from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

import torch


# Default export/evaluation modes for ISAR observation comparison.
DEFAULT_ISAR_OBSERVATION_MODES = ("identity", "log1p", "db_radar")


_COMMON_CONTEXT_DEFAULTS = {
    "normalization_mode": "none",
    "range_axis": None,
    "cross_range_axis": None,
    "future_physical_operator_name": None,
    "clamp_min": None,
    "clamp_max": None,
}


_MODE_CONTEXT_DEFAULTS = {
    "identity": {},
    "log1p": {
        "epsilon": 0.0,
        "normalization_mode": "none",
    },
    "db_radar": {
        "epsilon": 1e-4,
        "noise_floor_db": -40.0,
        "dynamic_range_db": 40.0,
        "normalization_mode": "db_floor_to_unit_interval",
        "clamp_min": 0.0,
        "clamp_max": None,
    },
    # Placeholder mode for forward compatibility when external physical operator arrives.
    "future_physical_operator": {
        "future_physical_operator_name": "placeholder",
    },
}


def to_isar_intensity(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert image-like tensors to a single-channel ISAR intensity map (1xHxW).
    """
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor with shape (C,H,W) or (H,W), got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        return tensor
    if tensor.shape[0] >= 3:
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return (tensor[:3] * weights).sum(dim=0, keepdim=True)
    return tensor.mean(dim=0, keepdim=True)


def _normalize_mode_name(obs_mode: str) -> str:
    if obs_mode is None:
        raise ValueError("obs_mode cannot be None")
    mode = str(obs_mode).strip().lower()
    if mode not in _MODE_CONTEXT_DEFAULTS:
        raise ValueError(f"Unknown isar_obs_mode: {obs_mode}")
    return mode


def _with_numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _apply_optional_clamp(tensor: torch.Tensor, context: Mapping[str, Any]) -> torch.Tensor:
    result = tensor
    clamp_min = _with_numeric_or_none(context.get("clamp_min"))
    clamp_max = _with_numeric_or_none(context.get("clamp_max"))

    if clamp_min is not None:
        result = torch.clamp_min(result, clamp_min)
    if clamp_max is not None:
        result = torch.clamp_max(result, clamp_max)
    return result


def resolve_isar_observation_context(obs_mode: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a fully-resolved observation context by merging defaults and caller overrides.
    """
    mode = _normalize_mode_name(obs_mode)

    resolved: dict[str, Any] = deepcopy(_COMMON_CONTEXT_DEFAULTS)
    resolved.update(deepcopy(_MODE_CONTEXT_DEFAULTS[mode]))
    if context:
        resolved.update(dict(context))

    resolved["mode"] = mode

    if mode == "log1p":
        eps = float(resolved.get("epsilon", 0.0))
        if eps < 0.0:
            raise ValueError(f"log1p epsilon must be >= 0, got {eps}")
        resolved["epsilon"] = eps

    if mode == "db_radar":
        noise_floor_db = float(resolved.get("noise_floor_db", -40.0))
        dynamic_range_db = float(resolved.get("dynamic_range_db", 40.0))
        epsilon = float(resolved.get("epsilon", 1e-4))

        if dynamic_range_db <= 0.0:
            raise ValueError(f"dynamic_range_db must be > 0, got {dynamic_range_db}")
        if epsilon <= 0.0:
            raise ValueError(f"db_radar epsilon must be > 0, got {epsilon}")

        resolved["noise_floor_db"] = noise_floor_db
        resolved["dynamic_range_db"] = dynamic_range_db
        resolved["epsilon"] = epsilon

    resolved["clamp_min"] = _with_numeric_or_none(resolved.get("clamp_min"))
    resolved["clamp_max"] = _with_numeric_or_none(resolved.get("clamp_max"))

    return resolved


def build_isar_observation_spec(obs_mode: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    mode = _normalize_mode_name(obs_mode)
    resolved_context = resolve_isar_observation_context(mode, context)
    return {
        "mode": mode,
        "context": resolved_context,
    }


def build_isar_observation_specs(
    obs_modes: Iterable[str],
    context_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    overrides = context_overrides or {}
    specs: list[dict[str, Any]] = []
    for mode in obs_modes:
        normalized = _normalize_mode_name(mode)
        specs.append(build_isar_observation_spec(normalized, overrides.get(normalized)))
    return specs


def build_default_isar_observation_specs(
    context_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return build_isar_observation_specs(DEFAULT_ISAR_OBSERVATION_MODES, context_overrides)


def apply_isar_observation_operator(
    tensor: torch.Tensor,
    obs_mode: str,
    context: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """
    Apply observation operator on render output or GT target.
    This interface is context-aware for future physical operator expansion.
    """
    mode = _normalize_mode_name(obs_mode)
    resolved = resolve_isar_observation_context(mode, context)

    if mode == "identity" or mode == "future_physical_operator":
        return _apply_optional_clamp(tensor, resolved)

    if mode == "log1p":
        epsilon = float(resolved.get("epsilon", 0.0))
        result = torch.log1p(torch.clamp_min(tensor, 0.0) + epsilon)
        return _apply_optional_clamp(result, resolved)

    if mode == "db_radar":
        epsilon = float(resolved["epsilon"])
        noise_floor_db = float(resolved["noise_floor_db"])
        dynamic_range_db = float(resolved["dynamic_range_db"])
        normalization_mode = str(resolved.get("normalization_mode", "db_floor_to_unit_interval"))

        db = 10.0 * torch.log10(torch.clamp_min(tensor, 0.0) + epsilon)
        if normalization_mode == "none":
            normalized = db
        elif normalization_mode == "db_floor_to_unit_interval":
            normalized = (db - noise_floor_db) / dynamic_range_db
        else:
            raise ValueError(
                f"Unsupported db_radar normalization_mode: {normalization_mode}. "
                "Supported: none | db_floor_to_unit_interval"
            )
        return _apply_optional_clamp(normalized, resolved)

    raise ValueError(f"Unknown isar_obs_mode: {obs_mode}")


def inverse_isar_observation_operator(
    tensor: torch.Tensor,
    obs_mode: str,
    context: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """
    Inverse observation operator to recover physical scatter intensity.
    """
    mode = _normalize_mode_name(obs_mode)
    resolved = resolve_isar_observation_context(mode, context)

    if mode == "identity" or mode == "future_physical_operator":
        return _apply_optional_clamp(tensor, resolved)

    if mode == "log1p":
        epsilon = float(resolved.get("epsilon", 0.0))
        restored = torch.expm1(tensor) - epsilon
        return torch.clamp_min(restored, 0.0)

    if mode == "db_radar":
        epsilon = float(resolved["epsilon"])
        noise_floor_db = float(resolved["noise_floor_db"])
        dynamic_range_db = float(resolved["dynamic_range_db"])
        normalization_mode = str(resolved.get("normalization_mode", "db_floor_to_unit_interval"))

        if normalization_mode == "none":
            db = tensor
        elif normalization_mode == "db_floor_to_unit_interval":
            db = tensor * dynamic_range_db + noise_floor_db
        else:
            raise ValueError(
                f"Unsupported db_radar normalization_mode: {normalization_mode}. "
                "Supported: none | db_floor_to_unit_interval"
            )

        restored = torch.pow(10.0, db / 10.0) - epsilon
        return torch.clamp_min(restored, 0.0)

    raise ValueError(f"Unknown isar_obs_mode: {obs_mode}")
