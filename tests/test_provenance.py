"""Tests for runtime provenance capture."""

from __future__ import annotations

import json

from site_safety_violation_detector.provenance import collect_environment_info


def test_environment_info_keys() -> None:
    info = collect_environment_info()
    expected = {
        "python_version",
        "python_executable",
        "platform",
        "machine",
        "processor",
        "system",
        "hostname",
        "cpu_count",
        "package_versions",
        "cuda",
        "mps",
    }
    assert expected.issubset(info.keys())


def test_package_versions_record_torch() -> None:
    info = collect_environment_info()
    # torch must be installed for this project to work; the version string
    # is non-empty when reported.
    versions = info["package_versions"]
    assert "torch" in versions
    assert versions["torch"] is None or isinstance(versions["torch"], str)


def test_environment_info_is_json_serialisable() -> None:
    info = collect_environment_info()
    blob = json.dumps(info)  # must not raise
    assert "package_versions" in blob


def test_cuda_block_has_available_flag() -> None:
    info = collect_environment_info()
    assert isinstance(info["cuda"], dict)
    assert "available" in info["cuda"]


def test_mps_block_has_available_flag() -> None:
    info = collect_environment_info()
    assert isinstance(info["mps"], dict)
    assert "available" in info["mps"]
