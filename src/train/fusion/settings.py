from pathlib import Path
from datetime import datetime

class BaseSettings:
    # Local Settings
    #base_path: Path = Path("/Users/wkong0407/Documents/yairi-lab/parker-preprocessing-comparison")
    # ABCI Settings
    base_path: Path = Path("/home/acf15412ed/parker-preprocessing-comparison")
    processed_path: Path = base_path / "processed_data"

    outputs_path: Path = base_path / "outputs"

class WISDM_Settings(BaseSettings):
    label_path: Path = BaseSettings.base_path / "raw_data" / "WISDM_segmented" / "labels"

    processed_path: Path = BaseSettings.base_path / "processed_data" / "wisdm"
    yaml_path: Path = BaseSettings.base_path / "wisdm.yaml"

    logs_dir: Path = BaseSettings.outputs_path / "wisdm_logs" / "fusion"

    exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    fusion1_dir: Path = logs_dir / "fusion1" / exec_time
    fusion2_dir: Path = logs_dir / "fusion2" / exec_time

    width = 200
    height = 200
    channels = 3

    patience = 10
    epochs = 200
    batch_size = 64

class Wheelchair_Settings(BaseSettings):
    label_path: Path = BaseSettings.base_path / "raw_data" / "車椅子_segmented" / "labels"

    processed_path: Path = BaseSettings.base_path / "processed_data" / "wheelchair"
    yaml_path: Path = BaseSettings.base_path / "車椅子.yaml"

    logs_dir: Path = BaseSettings.outputs_path / "wheelchair_logs" / "fusion"

    exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    fusion1_dir: Path = logs_dir / "fusion1" / exec_time
    fusion2_dir: Path = logs_dir / "fusion2" / exec_time

    width = 64
    height = 64
    channels = 3

    patience = 10
    epochs = 10000
    batch_size = 64


def get_wisdm_settings():
    return WISDM_Settings

def get_wheelchair_settings():
    return Wheelchair_Settings