"""Local model registry for GRADE inference.

The registry deliberately contains only the locally shipped paper checkpoint.
It has no machine-learning framework dependencies so command-line validation can
run in a lightweight Python environment.
"""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ID = "GRADE-paper"


@dataclass(frozen=True)
class ModelSpec:
    """Description of one supported local model checkpoint."""

    model_id: str
    directory_name: str
    description: str

    @property
    def local_dir(self) -> Path:
        """Return the repository-relative directory for this model."""
        return Path("models") / self.directory_name


MODEL_REGISTRY: dict[str, ModelSpec] = {
    DEFAULT_MODEL_ID: ModelSpec(
        model_id=DEFAULT_MODEL_ID,
        directory_name="grade-paper",
        description="GRADE paper checkpoint",
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    """Return a registered model specification.

    ``bool`` is rejected along with all other non-string values because model
    identifiers are textual CLI/API values.
    """
    if not isinstance(model_id, str):
        raise ValueError(f"Unknown model identifier: {model_id!r}")
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model identifier: {model_id!r}") from exc


def available_models() -> tuple[str, ...]:
    """Return registered model identifiers in registry order."""
    return tuple(MODEL_REGISTRY)
