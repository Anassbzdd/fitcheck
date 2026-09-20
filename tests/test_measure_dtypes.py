"""The measurement harness must measure the dtypes its rows are labelled with.

`scripts/measure.py` used to reach `--optimizer-dtype fp32` by casting every trainable
parameter to FP32 in place. Under LoRA that is a no-op -- peft already holds the
adapters in FP32 -- but under `--no-lora` every parameter is trainable, so the cast
rewrote the whole model and an fp16-labelled row actually measured an FP32 model, FP32
gradients and FP32 activations (fix.md problem 16).

These run on the CPU: nothing here needs CUDA, because the defect was in how the
optimizer was built, not in how memory was read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_MEASURE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "measure.py"


def _load_measure() -> Any:
    spec = importlib.util.spec_from_file_location("measure", _MEASURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through sys.modules.
    sys.modules["measure"] = module
    spec.loader.exec_module(module)
    return module


measure = _load_measure()


class _DecoderLayer(torch.nn.Module):
    """Named to match what ActivationDtypeProbe hooks on real causal LMs."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class _TinyModel(torch.nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.layer = _DecoderLayer(width)
        self.lm_head = torch.nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.layer(hidden))


def _full_ft_model(dtype=torch.float16) -> _TinyModel:
    return _TinyModel().to(dtype)


def _lora_like_model() -> _TinyModel:
    """An fp16 base with FP32 trainable adapters, the way peft leaves a LoRA model."""
    model = _TinyModel().to(torch.float16)
    for param in model.parameters():
        param.requires_grad_(False)
    model.lora_A = torch.nn.Parameter(torch.zeros(4, 8, dtype=torch.float32))
    return model


def _args(*argv: str) -> Any:
    return measure.parse_args(["fake/model", *argv])


# ---------------------------------------------------------------------------------
# The defect itself: full fine-tuning must not be silently rewritten to FP32
# ---------------------------------------------------------------------------------


def test_full_finetune_keeps_the_requested_compute_dtype() -> None:
    model = _full_ft_model()
    optimizer = measure.build_optimizer(model, _args("--no-lora", "--precision", "fp16"))

    assert isinstance(optimizer, measure.MasterWeightOptimizer)
    assert {p.dtype for p in model.parameters()} == {torch.float16}


def test_full_finetune_holds_the_master_copy_in_fp32() -> None:
    model = _full_ft_model()
    optimizer = measure.build_optimizer(model, _args("--no-lora", "--precision", "fp16"))

    assert {master.dtype for master in optimizer.masters} == {torch.float32}

    master_mib, master_dtype = measure.observed_master_weight_bytes(optimizer)
    trainable = sum(p.numel() for p in model.parameters())
    assert master_dtype == "fp32"
    assert master_mib == pytest.approx(trainable * 4 / measure.MIB)


def test_full_finetune_at_fp32_keeps_no_master_copy() -> None:
    """The parameters already are FP32, so there is nothing to shadow.

    Same condition fitcheck bills on (SPEC Component 3): billing the copy here would
    double-count the 4 bytes/param that W_base has already paid.
    """
    model = _full_ft_model(torch.float32)
    optimizer = measure.build_optimizer(model, _args("--no-lora", "--precision", "fp32"))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert measure.observed_master_weight_bytes(optimizer) == (0.0, "none")


def test_lora_path_is_unchanged() -> None:
    """The old in-place cast was harmless here, and every archived row is LoRA."""
    model = _lora_like_model()
    optimizer = measure.build_optimizer(model, _args("--precision", "fp16"))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert measure.observed_master_weight_bytes(optimizer) == (0.0, "none")
    assert model.layer.proj.weight.dtype == torch.float16
    assert model.lora_A.dtype == torch.float32


def test_master_step_trains_the_fp16_weights_with_fp32_state() -> None:
    model = _full_ft_model()
    optimizer = measure.build_optimizer(model, _args("--no-lora", "--precision", "fp16"))
    before = model.layer.proj.weight.detach().clone()

    optimizer.zero_grad()
    model(torch.ones(2, 8, dtype=torch.float16)).sum().backward()
    grad_mib, grad_dtype = measure.observed_gradient_bytes(model)
    optimizer.step()

    assert grad_dtype == "fp16"
    assert grad_mib > 0
    assert model.layer.proj.weight.dtype == torch.float16
    assert not torch.equal(model.layer.proj.weight, before)
    assert measure.observed_optimizer_state_bytes(optimizer)[1] == "fp32"


# ---------------------------------------------------------------------------------
# Observation and rejection
# ---------------------------------------------------------------------------------


def test_parameter_dtypes_separate_base_from_adapters() -> None:
    base, trainable = measure.observed_parameter_dtypes(_lora_like_model())
    assert (base, trainable) == ("fp16", "fp32")


def test_activation_probe_reads_the_hidden_state_dtype() -> None:
    model = _full_ft_model()
    probe = measure.ActivationDtypeProbe(model)
    model(torch.ones(2, 8, dtype=torch.float16))
    assert probe.remove() == "fp16"


def _measurement(**overrides: Any) -> Any:
    fields = {
        "cuda_context_mib": 140.875,
        "cuda_context_init_mib": 140.875,
        "after_load_allocated_mib": 100.0,
        "resident_before_step_mib": 100.0,
        "peak_allocated_mib": 200.0,
        "peak_reserved_mib": 220.0,
        "process_mib": 360.875,
        "peak_forward_mib": 150.0,
        "peak_backward_mib": 200.0,
        "peak_optimizer_mib": 180.0,
        "optimizer_state_mib": 16.0,
        "optimizer_state_dtype": "fp32",
        "master_weight_mib": 8.0,
        "master_weight_dtype": "fp32",
        "gradient_mib": 4.0,
        "gradient_dtype": "fp16",
        "base_param_dtype": "fp16",
        "trainable_param_dtype": "fp16",
        "activation_dtype": "fp16",
        "sdpa_backend": "n/a (not sdpa)",
        "trainable_params": 1000,
        "total_params": 1000,
        "step_seconds": 1.0,
        "weight_breakdown": {},
    }
    fields.update(overrides)
    return measure.Measurement(**fields)


def test_a_matching_row_is_accepted() -> None:
    args = _args("--no-lora", "--precision", "fp16")
    assert measure.dtype_mismatches(args, _measurement()) == []
    measure.reject_mislabelled_dtypes(args, _measurement())


def test_an_fp32_run_wearing_an_fp16_label_is_rejected() -> None:
    """Exactly what the old in-place cast produced under --no-lora."""
    args = _args("--no-lora", "--precision", "fp16")
    upcast = _measurement(
        base_param_dtype="fp32",
        trainable_param_dtype="fp32",
        gradient_dtype="fp32",
        activation_dtype="fp32",
        master_weight_mib=0.0,
        master_weight_dtype="none",
    )

    with pytest.raises(SystemExit) as excinfo:
        measure.reject_mislabelled_dtypes(args, upcast)

    message = str(excinfo.value)
    for field in (
        "base_param_dtype",
        "trainable_param_dtype",
        "gradient_dtype",
        "activation_dtype",
    ):
        assert field in message


def test_a_quantized_base_is_not_called_mislabelled_for_its_fp32_upcast() -> None:
    """peft's kbit prep upcasts norms, embeddings and the LM head on purpose.

    That upcast is measured and already lives in `memory/activations._PROFILES`, so
    checking the base and activation families against --precision here would reject
    every QLoRA row in the archive.
    """
    args = _args("--qlora", "--precision", "fp16")
    qlora = _measurement(
        base_param_dtype="fp32",
        trainable_param_dtype="fp32",
        gradient_dtype="fp32",
        activation_dtype="fp32",
    )
    assert measure.dtype_mismatches(args, qlora) == []


def test_lora_gradients_must_be_fp32() -> None:
    args = _args("--precision", "fp16")
    half = _measurement(trainable_param_dtype="fp32", gradient_dtype="fp16")
    assert "gradient_dtype" in "".join(measure.dtype_mismatches(args, half))


# ---------------------------------------------------------------------------------
# The harness and fitcheck must split full fine-tuning the same way
# ---------------------------------------------------------------------------------


def _bytes_per_param(model, optimizer, params: int) -> dict[str, float]:
    def live(*tensors: Any) -> int:
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    weights = live(*model.parameters())
    masters = live(*getattr(optimizer, "masters", []))

    optimizer.zero_grad()
    model(torch.ones(4, 8, dtype=model.layer.proj.weight.dtype)).sum().backward()
    gradients = live(*[p.grad for p in model.parameters()])
    optimizer.step()
    states = measure.observed_optimizer_state_bytes(optimizer)[0] * measure.MIB

    return {
        "W_base": weights / params,
        "S_optim": (masters + states) / params,
        "G_grad": gradients / params,
    }


@pytest.mark.parametrize(
    ("precision", "dtype", "expected"),
    [
        ("fp16", torch.float16, {"W_base": 2.0, "S_optim": 12.0, "G_grad": 2.0}),
        ("fp32", torch.float32, {"W_base": 4.0, "S_optim": 8.0, "G_grad": 4.0}),
    ],
)
def test_full_finetune_matches_fitchecks_own_split(
    precision: str, dtype: Any, expected: dict[str, float]
) -> None:
    """SPEC Component 3's invariant table, measured rather than asserted on paper.

    Both paths land at 16 bytes/param; what the fix changed is that the harness now
    puts each of those bytes where fitcheck puts them, instead of reaching the same
    total with the master copy folded into the weights.
    """
    from fitcheck.memory.gradients import estimate_gradient_memory
    from fitcheck.memory.optimizer import estimate_optimizer_memory
    from fitcheck.memory.weights import estimate_weight_memory

    model = _full_ft_model(dtype)
    params = sum(p.numel() for p in model.parameters())
    args = _args("--no-lora", "--precision", precision)

    observed = _bytes_per_param(model, measure.build_optimizer(model, args), params)
    assert observed == expected
    assert sum(observed.values()) == 16.0

    scale = measure.MIB / params
    predicted = {
        "W_base": estimate_weight_memory(params, precision) * scale,
        "S_optim": estimate_optimizer_memory(params, "adamw", False, "fp32", precision)
        * scale,
        "G_grad": estimate_gradient_memory(params, precision) * scale,
    }
    assert predicted == expected
