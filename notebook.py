"""Pseudo-notebook script describing the inference pipeline.

This file is organised in notebook-like "cells" separated by long dashed
comment blocks so it can be executed as a regular Python script while still
highlighting each conceptual step of the inference workflow.  The pipeline is
constructed from the components available in the repository and the
requirements described in ``prd.md``.
"""

# ----------------------------------------------------------------------------------
# Cell 1: Imports & Environment Setup
# ----------------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torchaudio
import yaml

from src import commons
from src.datasets.base_dataset import apply_preprocessing
from src.models import models

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "whisper_frontend_specrnet.yaml"
DEFAULT_THRESHOLD = 0.5


# ----------------------------------------------------------------------------------
# Cell 2: Device & Configuration Utilities
# ----------------------------------------------------------------------------------
def select_device(force_cpu: bool = False) -> torch.device:
    """Return the torch device that should be used for inference."""

    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(path_like: Optional[Union[str, Path]]) -> Optional[Path]:
    """Resolve string or Path inputs relative to the project root."""

    if path_like in (None, "", "None"):
        return None

    resolved_path = Path(path_like)
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()
    return resolved_path


def load_yaml_config(config_path: Union[str, Path]) -> Dict:
    """Load a YAML configuration file describing the model to be used."""

    config_path = resolve_path(config_path)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            f"Could not locate configuration file at: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config: Dict = yaml.safe_load(handle)
    return config


# ----------------------------------------------------------------------------------
# Cell 3: Audio Loading & Pre-processing
# ----------------------------------------------------------------------------------
def load_audio_tensor(audio_path: Union[str, Path], device: torch.device) -> torch.Tensor:
    """Load and pre-process an audio file ready for the neural network.

    Steps follow the PRD specification and dataset utilities:
    - Load and normalise the waveform.
    - Apply resampling, mono conversion, silence trimming, and padding to
      obtain a fixed 30 second / 480 000 sample clip.
    - Return a tensor with shape ``(1, 480000)`` ready for batching.
    """

    audio_path = resolve_path(audio_path)
    if audio_path is None or not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = torchaudio.load(str(audio_path), normalize=True)
    waveform, _ = apply_preprocessing(waveform, sample_rate)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # add batch dimension

    return waveform.float().to(device)


# ----------------------------------------------------------------------------------
# Cell 4: Model Construction & Checkpoint Loading
# ----------------------------------------------------------------------------------
def build_model_from_config(model_section: Dict, device: torch.device) -> torch.nn.Module:
    """Instantiate the requested model architecture and move it to the device."""

    model_name = model_section.get("name")
    if model_name is None:
        raise ValueError("Model configuration must include a 'name' key.")

    parameters = model_section.get("parameters", {})
    device_str = device.type if isinstance(device, torch.device) else str(device)
    model = models.get_model(model_name=model_name, config=parameters, device=device_str)
    model = model.to(device)
    model.eval()
    return model


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Optional[Union[str, Path]],
    device: torch.device,
) -> torch.nn.Module:
    """Load a checkpoint if one is provided and return the ready model."""

    resolved_checkpoint = resolve_path(checkpoint_path)
    if resolved_checkpoint is None:
        raise ValueError(
            "A trained checkpoint is required for inference but none was provided."
        )

    if not resolved_checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at: {resolved_checkpoint}"
        )

    state_dict = torch.load(str(resolved_checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----------------------------------------------------------------------------------
# Cell 5: End-to-End Pipeline Preparation
# ----------------------------------------------------------------------------------
def prepare_inference_pipeline(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    checkpoint_path: Optional[Union[str, Path]] = None,
    force_cpu: bool = False,
) -> Tuple[torch.nn.Module, torch.device, float]:
    """Prepare the inference model, device, and decision threshold.

    Returns a tuple ``(model, device, threshold)`` that can be re-used across
    multiple predictions without rebuilding the network every time.
    """

    config = load_yaml_config(config_path)

    # Seed everything for reproducibility
    seed = config.get("data", {}).get("seed", 42)
    commons.set_seed(seed)

    device = select_device(force_cpu=force_cpu)
    model = build_model_from_config(config.get("model", {}), device)

    # Allow direct argument override of the checkpoint path; if not provided
    # fall back to the one listed in the configuration file.
    checkpoint_override = checkpoint_path or config.get("checkpoint", {}).get("path")
    model = load_model_weights(model, checkpoint_override, device)

    threshold = config.get("model", {}).get("parameters", {}).get(
        "decision_threshold", DEFAULT_THRESHOLD
    )
    return model, device, threshold


# ----------------------------------------------------------------------------------
# Cell 6: Prediction Helpers
# ----------------------------------------------------------------------------------
def predict_deepfake_probability(
    model: torch.nn.Module,
    audio_tensor: torch.Tensor,
    device: torch.device,
) -> float:
    """Return the model's probability score for the bonafide (real) class."""

    model.eval()
    with torch.no_grad():
        logits = model(audio_tensor.to(device))
        probabilities = torch.sigmoid(logits)
    return probabilities.squeeze().item()


def label_from_probability(probability: float, threshold: float) -> str:
    """Convert a probability into a human readable label."""

    return "real" if probability >= threshold else "fake"


# ----------------------------------------------------------------------------------
# Cell 7: User-Facing Convenience Function
# ----------------------------------------------------------------------------------
def classify_audio_file(
    audio_path: Union[str, Path],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    checkpoint_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    force_cpu: bool = False,
) -> str:
    """Classify an audio file as ``"real"`` or ``"fake"``.

    Parameters
    ----------
    audio_path:
        Path to the ``.wav`` file that should be analysed.
    config_path:
        YAML file describing the model architecture and default threshold.
    checkpoint_path:
        Optional path to the trained weights. If omitted, the value from the
        configuration file is used.
    threshold:
        Decision boundary for mapping probabilities to labels. Defaults to the
        configuration threshold (or ``0.5`` if unspecified).
    force_cpu:
        If ``True`` forces CPU inference even when a GPU is available.
    """

    model, device, default_threshold = prepare_inference_pipeline(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        force_cpu=force_cpu,
    )

    audio_tensor = load_audio_tensor(audio_path, device)
    probability = predict_deepfake_probability(model, audio_tensor, device)

    decision_threshold = default_threshold if threshold is None else threshold
    return label_from_probability(probability, decision_threshold)


# ----------------------------------------------------------------------------------
# Cell 8: Example Usage (optional manual execution)
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "This script is intended to be imported and used through the "
        "classify_audio_file function."
    )
    print(
        "Example: label = classify_audio_file('sample.wav', 'config.yaml', 'model.pth')"
    )
