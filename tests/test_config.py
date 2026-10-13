"""Tests for R3AF config."""

import os
import tempfile
import pytest

from r3af.models.config import R3AFConfig


class TestR3AFConfig:
    def test_default_values(self):
        config = R3AFConfig()
        assert config.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.batch_size == 2
        assert config.group_size == 4
        assert config.lambda_tcr == 0.1
        assert config.benign_ratio == 0.15

    def test_yaml_roundtrip(self):
        config = R3AFConfig(
            model_name="test-model",
            batch_size=8,
            lambda_tcr=0.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.yaml")
            config.to_yaml(path)

            loaded = R3AFConfig.from_yaml(path)
            assert loaded.model_name == "test-model"
            assert loaded.batch_size == 8
            assert loaded.lambda_tcr == 0.5

    def test_get_reward_weights(self):
        config = R3AFConfig()
        weights = config.get_reward_weights()
        assert "R_prm" in weights
        assert "R_anchor" in weights
        assert "R_corrigible" in weights
        assert "R_ans" in weights
        assert "R_len" in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_from_yaml_with_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.yaml")
            config = R3AFConfig(model_name="override-model")
            config.to_yaml(path)

            loaded = R3AFConfig.from_yaml(path)
            assert loaded.model_name == "override-model"
