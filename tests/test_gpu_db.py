from __future__ import annotations
import pytest
from fitcheck.gpu_db import get_gpu


@pytest.mark.parametrize(
    ("gpu_name", "display_name", "vram_mib", "usable_mib"),
    [
        ("t4", "Tesla T4", 16_384, 15_360),
        ("4090", "RTX 4090", 24_576, 23_500),
        ("a100-80", "A100 80GB", 81_920, 79_000),
        ("3090", "RTX 3090", 24_576, 23_500),
        ("a100-40", "A100 40GB", 40_960, 39_500),
        ("h100", "H100 80GB", 81_920, 79_000),
        ("l4", "L4 24GB", 24_576, 23_000),
        ("3060-12", "RTX 3060 12GB", 12_288, 11_500),
        ("4070ti", "RTX 4070 Ti", 12_288, 11_500),
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


@pytest.mark.parametrize("gpu_name", ["T4", " t4 ", "T4  ", "  T4"])
def test_get_gpu_name_lookup_is_case_and_whitespace_insensitive(gpu_name: str) -> None:
    """GPU names normalize via strip + casefold before dict lookup."""
    gpu = get_gpu(gpu_name)

    assert gpu.name == "Tesla T4"


def test_get_gpu_unknown_name_has_actionable_error() -> None:
    """Unknown aliases explain the issue and list valid choices."""
    with pytest.raises(ValueError, match="Unknown GPU 'not-a-gpu'") as error:
        get_gpu("not-a-gpu")

    assert "Available GPUs:" in str(error.value)


def test_get_gpu_vram_override_returns_custom_spec() -> None:
    """SPEC.md's --vram-mib escape hatch for unlisted GPUs."""
    gpu = get_gpu(vram_mib=24_000)

    assert gpu.name == "Custom GPU"
    assert gpu.vram_mib == 24_000
    assert gpu.usable_mib == 22_800  # 24_000 * 95 // 100


def test_get_gpu_vram_override_with_custom_name() -> None:
    gpu = get_gpu(name="My Rig", vram_mib=10_000)

    assert gpu.name == "My Rig"
    assert gpu.vram_mib == 10_000
    assert gpu.usable_mib == 9_500


def test_get_gpu_vram_override_ignores_blank_name() -> None:
    """A blank/whitespace-only name falls back to the default custom label."""
    gpu = get_gpu(name="   ", vram_mib=10_000)

    assert gpu.name == "Custom GPU"


@pytest.mark.parametrize("bad_vram", [0, -1, True, 12.5, "24000"])
def test_get_gpu_rejects_invalid_vram_override(bad_vram: object) -> None:
    with pytest.raises(ValueError, match="vram_mib must be a positive integer"):
        get_gpu(vram_mib=bad_vram)  # type: ignore[arg-type]


def test_get_gpu_requires_name_or_vram_override() -> None:
    with pytest.raises(ValueError, match="Provide a GPU name or a positive vram_mib override"):
        get_gpu()
