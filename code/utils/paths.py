"""Paths relative to this project, independent of the working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
DATA_ROOT = PROJECT_ROOT / "datasets"
PUNET_ROOT = DATA_ROOT / "PUNet"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_ROOT / "stage2_latest.pth"


def writable_output(path):
    """Keep generated results outside bundled code, data and pretrained weights."""
    path = Path(path).resolve()
    for protected in (CODE_ROOT, DATA_ROOT, CHECKPOINT_ROOT):
        if path == protected or protected in path.parents:
            raise ValueError("Choose an output outside the bundled directory: %s" % protected)
    return path
