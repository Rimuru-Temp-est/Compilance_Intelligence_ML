"""Environment-backed project paths for local backend development."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_root: Path = _path_from_env("DATA_ROOT", PROJECT_ROOT / "Datasets")
    aml_model_path: Path = _path_from_env(
        "AML_MODEL_PATH",
        PROJECT_ROOT / "AML_Model" / "processed" / "aml_model" / "xgb_behavioral_model.pkl",
    )
    aml_metadata_path: Path = _path_from_env(
        "AML_METADATA_PATH",
        PROJECT_ROOT / "AML_Model" / "processed" / "aml_model" / "metadata.json",
    )
    master_table_root: Path = _path_from_env(
        "MASTER_TABLE_ROOT", PROJECT_ROOT / "backend_data" / "master_table"
    )
    behavior_history_path: Path = _path_from_env(
        "BEHAVIOR_HISTORY_PATH", PROJECT_ROOT / "backend_data" / "processed" / "behavior_history.npz"
    )


settings = Settings()
