"""vLLM-based model wrappers for fast batched inference.

Contains:
- VLLMModel: Direct vLLM wrapper for evaluation (loads in-process).
- VLLMGenerationWorker: Multiprocessing wrapper that runs vLLM in a
  separate child process with its own CUDA_VISIBLE_DEVICES, enabling
  GPU-isolated generation during training.

Usage (VLLMModel — evaluation):
    from r3af.models.vllm_model import VLLMModel
    model = VLLMModel(config)
    responses = model.generate_batch(["What is 2+2?"])

Usage (VLLMGenerationWorker — training):
    from r3af.models.vllm_model import VLLMGenerationWorker
    worker = VLLMGenerationWorker(config, gpu_id=3)
    responses = worker.generate_batch(["What is 2+2?"])
    worker.update_lora("/path/to/adapter")
    worker.shutdown()
"""

import atexit
import logging
import os
import traceback
from typing import List, Optional

import torch.multiprocessing as mp

from .config import R3AFConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function (runs in child process with its own CUDA_VISIBLE_DEVICES)
# ---------------------------------------------------------------------------

def _vllm_worker_fn(
    gpu_id: int,
    model_name: str,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    lora_r: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    gpu_memory_utilization: float = 0.35,
):
    """Entry point for the vLLM child process.

    Protocol (via request_queue):
        ("generate", List[str])     -> response_queue <- List[str]
        ("load_lora", str)          -> response_queue <- "ok" | error str
        None                        -> worker exits
    """
    # Isolate to the designated GPU before importing vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    _log = _logging.getLogger("vllm_worker")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        _log.info(f"vLLM worker starting on GPU {gpu_id}, model={model_name}")

        llm = LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            enable_lora=True,
            max_lora_rank=max(lora_r, 64),
            gpu_memory_utilization=gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        _log.info("vLLM worker ready")
        response_queue.put("ready")

        # Current LoRA state
        lora_request: Optional[LoRARequest] = None
        lora_counter = 0  # unique id for each LoRA load

        while True:
            msg = request_queue.get()
            if msg is None:
                _log.info("vLLM worker shutting down")
                break

            cmd, payload = msg

            if cmd == "generate":
                try:
                    outputs = llm.generate(
                        payload,
                        sampling_params,
                        lora_request=lora_request,
                    )
                    texts = [o.outputs[0].text for o in outputs]
                    response_queue.put(("ok", texts))
                except Exception as e:
                    _log.error(f"Generate error: {e}\n{traceback.format_exc()}")
                    response_queue.put(("error", str(e)))

            elif cmd == "load_lora":
                try:
                    adapter_path = payload
                    lora_counter += 1
                    lora_request = LoRARequest(
                        lora_name=f"adapter_{lora_counter}",
                        lora_int_id=lora_counter,
                        lora_path=adapter_path,
                    )
                    _log.info(f"LoRA adapter loaded: {adapter_path}")
                    response_queue.put(("ok", "lora_loaded"))
                except Exception as e:
                    _log.error(f"LoRA load error: {e}\n{traceback.format_exc()}")
                    response_queue.put(("error", str(e)))

    except Exception as e:
        _log.error(f"vLLM worker fatal error: {e}\n{traceback.format_exc()}")
        response_queue.put(("fatal", str(e)))


class VLLMGenerationWorker:
    """Manages a vLLM engine in a separate process for GPU-isolated generation.

    The child process sets CUDA_VISIBLE_DEVICES to the specified gpu_id,
    so the vLLM engine uses a different GPU from the training process.
    Communication happens via mp.Queue with string lists (negligible overhead).
    """

    def __init__(self, config: R3AFConfig, gpu_id: int = 3, timeout: int = 600):
        self.config = config
        self.gpu_id = gpu_id
        self.timeout = timeout
        self._alive = False

        ctx = mp.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()

        self._process = ctx.Process(
            target=_vllm_worker_fn,
            kwargs=dict(
                gpu_id=gpu_id,
                model_name=config.model_name,
                max_model_len=config.max_seq_length,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                lora_r=config.lora_r,
                request_queue=self._request_queue,
                response_queue=self._response_queue,
                gpu_memory_utilization=getattr(config, "vllm_gpu_memory_utilization", 0.35),
            ),
            daemon=False,  # vLLM internally spawns child processes
        )
        self._process.start()

        # Wait for worker to be ready
        logger.info(f"Waiting for vLLM worker to start on GPU {gpu_id}...")
        ready_msg = self._response_queue.get(timeout=self.timeout)
        if ready_msg == "ready":
            self._alive = True
            atexit.register(self.shutdown)
            logger.info("vLLM worker is ready")
        else:
            raise RuntimeError(f"vLLM worker failed to start: {ready_msg}")

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Send prompts to the worker and return generated texts."""
        if not self._alive:
            raise RuntimeError("vLLM worker is not alive")
        self._request_queue.put(("generate", prompts))
        status, result = self._response_queue.get(timeout=self.timeout)
        if status == "ok":
            return result
        raise RuntimeError(f"vLLM generation failed: {result}")

    def update_lora(self, adapter_path: str):
        """Tell the worker to load a new LoRA adapter."""
        if not self._alive:
            raise RuntimeError("vLLM worker is not alive")
        self._request_queue.put(("load_lora", adapter_path))
        status, result = self._response_queue.get(timeout=self.timeout)
        if status == "ok":
            logger.info(f"vLLM worker loaded LoRA from {adapter_path}")
        else:
            raise RuntimeError(f"vLLM LoRA load failed: {result}")

    def shutdown(self):
        """Cleanly stop the worker process and its children (vLLM EngineCore)."""
        if self._alive:
            self._request_queue.put(None)
            self._process.join(timeout=30)
            if self._process.is_alive():
                # Kill the process tree (vLLM spawns child processes)
                import signal
                try:
                    os.kill(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self._process.join(timeout=10)
                if self._process.is_alive():
                    self._process.kill()
            self._alive = False
            logger.info("vLLM worker shut down")

    def __del__(self):
        self.shutdown()


class VLLMModel:
    """vLLM model wrapper for fast batched inference.

    Loads an unquantized bf16 model via the vLLM LLM engine and provides
    batched generation. Keeps a HuggingFace tokenizer for chat template
    application and token counting.

    Note: vLLM does not support BitsAndBytes 4-bit quantization. The model
    is loaded in bf16, which fits on a single A6000 (~14GB weights + ~10GB
    KV cache for Qwen2.5-7B).
    """

    def __init__(self, config: R3AFConfig, tensor_parallel_size: int = 1):
        """Initialize vLLM model.

        Args:
            config: R3AFConfig with model_name, max_seq_length, and
                generation parameters.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
        """
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.config = config

        logger.info(f"Loading model via vLLM: {config.model_name}")
        self.llm = LLM(
            model=config.model_name,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=config.max_seq_length,
            gpu_memory_utilization=0.6,
            enforce_eager=True,
        )

        # Keep HF tokenizer for chat template + token counting
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        logger.info("vLLM model loaded successfully")

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Batch generation — the key speedup over sequential HF generate.

        Args:
            prompts: List of fully-formatted prompt strings (with chat
                template already applied).

        Returns:
            List of generated response texts (one per prompt).
        """
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]

    def generate_single(self, prompt: str) -> str:
        """Single prompt generation (for compatibility).

        Args:
            prompt: Fully-formatted prompt string.

        Returns:
            Generated response text.
        """
        return self.generate_batch([prompt])[0]
