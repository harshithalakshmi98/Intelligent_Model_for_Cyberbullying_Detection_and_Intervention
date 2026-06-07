from pathlib import Path


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "datasets").is_dir() and (parent / "requirements.txt").is_file():
            return parent
    return here.parents[2]


PROJECT_ROOT = get_project_root()
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
