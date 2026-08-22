"""Tests for the built-in GPU specification database."""

from __future__ import annotations

import pytest

from fitcheck.gpu_db import get_gpu


@pytest.mark.parametrize(
    ("gpu_name", "display_name", "vram_mib", "usable_mib"),
    [
        ("t4", "Tesla T4", 16_384, 15_360),
        ("4090", "RTX 4090", 24_576, 23_500),
        ("a100-80", "A100 80GB", 81_920, 79_000),
    ],
)
def test_get_gpu_returns_known_gpu_spec(
    gpu_name: str,
    display_name: str,
    vram_mib: int,
    usable_mib: int,
) -> None:
    """Known aliases resolve to their documented memory specifications."""
    gpu = get_gpu(gpu_name)

    assert gpu.name == display_name
    assert gpu.vram_mib == vram_mib
    assert gpu.usable_mib == usable_mib


def test_get_gpu_unknown_name_has_actionable_error() -> None:
    """Unknown aliases explain the issue and list valid choices."""
    with pytest.raises(ValueError, match="Unknown GPU 'not-a-gpu'") as error:
        get_gpu("not-a-gpu")

    assert "Available GPUs:" in str(error.value)
