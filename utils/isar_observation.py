import torch

def apply_isar_observation_operator(tensor: torch.Tensor, obs_mode: str) -> torch.Tensor:
    """
    Apply observation operator on either render output or GT target.
    tensor: (C, H, W) or (H, W), usually non-negative real values (scatter intensity)
    obs_mode: "identity" | "log1p" | "future_physical_operator"
    
    NOTE: As of Phase "Non-Coupled Observation API", this operator is decoupled from the 
    training loss loop to preserve 3DGS high-gradient geometry split/clone dynamics.
    It serves strictly as a forward evaluation, comparison, and export projection interface 
    for observation domain mapping.
    """
    mode = obs_mode.lower()
    if mode == "identity":
        return tensor
    elif mode == "log1p":
        return torch.log1p(tensor)
    elif mode == "future_physical_operator":
        # Placeholder for complex envelope / synthetic aperture processing integration
        # Needs point-wise phase shifts / sinc interpolation or matched filtering behaviors in future
        return tensor 
    else:
        raise ValueError(f"Unknown isar_obs_mode: {obs_mode}")

def inverse_isar_observation_operator(tensor: torch.Tensor, obs_mode: str) -> torch.Tensor:
    """
    Inverse operator to recover physical scatter intensity.
    """
    mode = obs_mode.lower()
    if mode == "identity":
        return tensor
    elif mode == "log1p":
        return torch.expm1(tensor)
    elif mode == "future_physical_operator":
        # Placeholder
        return tensor
    else:
        raise ValueError(f"Unknown isar_obs_mode: {obs_mode}")
