#!/usr/bin/env python3
"""
Pytest tests for deployment policy functionality.
Tests that deployment policies correctly handle model instantiation and device management.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from data import ActivationDataModule
from data.deployment_policy import (
    CPUOnlyPolicy,
    DeploymentPolicy,
    DynamicPolicy,
    GPUOnlyPolicy,
    create_deployment_policy,
)


class TestDeploymentPolicyEnum:
    """Test the DeploymentPolicy enum."""

    def test_policy_creation_from_string(self):
        """Test creating policies from string values."""
        assert DeploymentPolicy.from_string("cpu_only") == DeploymentPolicy.CPU_ONLY
        assert DeploymentPolicy.from_string("gpu_only") == DeploymentPolicy.GPU_ONLY
        assert DeploymentPolicy.from_string("dynamic") == DeploymentPolicy.DYNAMIC

    def test_invalid_policy_string(self):
        """Test that invalid policy strings raise ValueError."""
        with pytest.raises(ValueError, match="Unknown deployment policy"):
            DeploymentPolicy.from_string("invalid_policy")


class TestActivationDataModuleIntegration:
    """Test deployment policy integration with ActivationDataModule."""

    def test_cpu_only_policy_integration(self):
        """Test CPU-only policy can be configured in ActivationDataModule."""
        datamodule = ActivationDataModule(deployment_policy="cpu_only", buffer_size=1000, batch_size=100)
        assert datamodule.deployment_policy == DeploymentPolicy.CPU_ONLY

    def test_gpu_only_policy_integration(self):
        """Test GPU-only policy can be configured in ActivationDataModule."""
        datamodule = ActivationDataModule(deployment_policy="gpu_only", buffer_size=1000, batch_size=100)
        assert datamodule.deployment_policy == DeploymentPolicy.GPU_ONLY

    def test_dynamic_policy_integration(self):
        """Test dynamic policy can be configured in ActivationDataModule."""
        datamodule = ActivationDataModule(deployment_policy="dynamic", buffer_size=1000, batch_size=100)
        assert datamodule.deployment_policy == DeploymentPolicy.DYNAMIC

    def test_default_policy_is_dynamic(self):
        """Test that default policy is dynamic."""
        datamodule = ActivationDataModule(buffer_size=1000, batch_size=100)
        assert datamodule.deployment_policy == DeploymentPolicy.DYNAMIC


class TestDeviceMapping:
    """Test device mapping functionality."""

    def test_gpu_only_device_parsing(self):
        """Test that GPU-only policy correctly parses device maps."""
        # Test single specific GPU
        policy = create_deployment_policy(DeploymentPolicy.GPU_ONLY, device_map="cuda:3")
        devices, is_multi_gpu = policy._parse_device_map()
        assert devices == ["cuda:3"]
        assert is_multi_gpu == False

        # Test multi-GPU
        policy = create_deployment_policy(DeploymentPolicy.GPU_ONLY, device_map="cuda:0,1,2,3")
        devices, is_multi_gpu = policy._parse_device_map()
        assert devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        assert is_multi_gpu == True

    def test_gpu_only_device_tracking_logic(self):
        """Test that GPU-only policy correctly calculates device tracking without loading models."""
        # Test the core logic without actually loading models
        policy = create_deployment_policy(DeploymentPolicy.GPU_ONLY, device_map="cuda:3")

        # Test device parsing
        devices, is_multi_gpu = policy._parse_device_map()
        gpu_device_map = devices[0] if not is_multi_gpu else policy.device_map

        # This is what setup_models should set current_device to
        expected_device = gpu_device_map
        assert expected_device == "cuda:3"

        # Manually set what setup_models would set to test select_device
        policy.current_device = gpu_device_map
        assert policy.select_device() == "cuda:3"

    def test_dynamic_policy_device_tracking_logic(self):
        """Test that dynamic policy correctly calculates GPU device for switching."""
        # Test the core logic without actually loading models
        policy = create_deployment_policy(DeploymentPolicy.DYNAMIC, device_map="cuda:3")

        # Test device parsing
        devices, is_multi_gpu = policy._parse_device_map()
        gpu_device_map = devices[0] if not is_multi_gpu else policy.device_map

        # Manually set what setup_models would set to test switching logic
        policy.gpu_device = gpu_device_map
        policy.current_device = "cpu"
        policy.gpu_model = Mock()  # Mock model to pass None checks

        # Test switching to GPU with low buffer
        buffer_stats = {"valid_percentage": 30.0}  # Below GPU threshold (50%)
        device = policy.select_device(buffer_stats)
        assert device == "cuda:3"
        assert policy.current_device == "cuda:3"


# Note: Additional tests for policy classes will be added once they're implemented
class TestDeploymentPolicyClasses:
    """Test the deployment policy behavior classes."""

    @pytest.fixture
    def mock_models(self):
        """Create mock CPU and GPU models for testing."""
        cpu_model = Mock()
        cpu_model.to.return_value = cpu_model
        gpu_model = Mock()
        gpu_model.to.return_value = gpu_model
        return cpu_model, gpu_model

    def test_cpu_only_policy_class_exists(self):
        """Test that CPUOnlyPolicy class can be imported."""
        # This will pass once we implement the class
        assert CPUOnlyPolicy is not None

    def test_gpu_only_policy_class_exists(self):
        """Test that GPUOnlyPolicy class can be imported."""
        # This will pass once we implement the class
        assert GPUOnlyPolicy is not None

    def test_dynamic_policy_class_exists(self):
        """Test that DynamicPolicy class can be imported."""
        # This will pass once we implement the class
        assert DynamicPolicy is not None


if __name__ == "__main__":
    pytest.main([__file__])
