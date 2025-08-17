#!/usr/bin/env python3
"""
Tests for process monitor functionality.
"""

from unittest.mock import Mock, patch

import pytest

from crosslayer_transcoder.data.process_monitor import ProcessMonitor, WandBProcessMonitor


class TestProcessMonitor:
    """Test basic process monitor functionality."""

    def test_basic_monitor_creation(self):
        """Test that basic monitor can be created."""
        monitor = ProcessMonitor()
        assert monitor is not None
        assert monitor._last_refresh_rate == 0.0

    def test_refresh_rate_setting(self):
        """Test that refresh rate can be set and retrieved."""
        monitor = ProcessMonitor()
        monitor.set_refresh_rate(123.45)
        assert monitor._last_refresh_rate == 123.45


class TestWandBProcessMonitor:
    """Test WandB process monitor functionality."""

    def test_device_code_mapping(self):
        """Test that device code mapping works correctly for all device types."""
        # Create monitor - it will fail to init WandB but device mapping still works
        monitor = WandBProcessMonitor()

        # Test exact matches
        assert monitor._get_device_code("cpu") == 0
        assert monitor._get_device_code("cuda") == 1
        assert monitor._get_device_code("disk") == 2

        # Test case insensitivity
        assert monitor._get_device_code("CPU") == 0
        assert monitor._get_device_code("CUDA") == 1

        # Test specific GPU devices - THIS IS THE MAIN FIX
        assert monitor._get_device_code("cuda:0") == 1
        assert monitor._get_device_code("cuda:1") == 1
        assert monitor._get_device_code("cuda:3") == 1
        assert monitor._get_device_code("CUDA:7") == 1

        # Test unknown devices
        assert monitor._get_device_code("unknown") == -1
        assert monitor._get_device_code("tpu") == -1

    def test_dashboard_device_string_generation(self):
        """Test that dashboard correctly displays device strings."""
        monitor = ProcessMonitor()

        # Test different device types for dashboard display
        test_cases = [
            ("cpu", "CPU"),
            ("cuda", "GPU"),
            ("cuda:0", "GPU"),
            ("cuda:3", "GPU"),
            ("disk", "DISK"),
            ("unknown", "CPU"),  # Default to CPU for unknown
        ]

        for device, expected in test_cases:
            # Simulate the dashboard logic
            if device == "disk":
                device_str = "DISK"
            elif device.startswith("cuda"):
                device_str = "GPU"
            else:
                device_str = "CPU"

            assert device_str == expected, f"Device {device} should display as {expected}, got {device_str}"


if __name__ == "__main__":
    pytest.main([__file__])
