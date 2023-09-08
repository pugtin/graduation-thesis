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

    logs_dir: Path = BaseSettings.outputs_path / "wisdm_logs" / "singlehead"

    exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    mtf_dir: Path = logs_dir / "mtf" / exec_time
    rp_dir: Path = logs_dir / "rp" / exec_time
    gasf_dir: Path = logs_dir / "gasf" / exec_time
    gadf_dir: Path = logs_dir / "gadf" / exec_time

    width = 64
    height = 64
    rgb_channels = 3
    grayscale_channels = 1

    patience = 10
    epochs = 200
    batch_size = 8

class Wheelchair_Settings(BaseSettings):
    label_path: Path = BaseSettings.base_path / "raw_data" / "車椅子_segmented" / "labels"

    processed_path: Path = BaseSettings.base_path / "processed_data" / "wheelchair"
    yaml_path: Path = BaseSettings.base_path / "車椅子.yaml"

    logs_dir: Path = BaseSettings.outputs_path / "wheelchair_logs" / "singlehead"

    exec_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    mtf_dir: Path = logs_dir / "mtf" / exec_time
    rp_dir: Path = logs_dir / "rp" / exec_time
    gasf_dir: Path = logs_dir / "gasf" / exec_time
    gadf_dir: Path = logs_dir / "gadf" / exec_time

    width = 128
    height = 128
    rgb_channels = 3
    grayscale_channels = 1

    patience = 10
    epochs = 10000
    batch_size = 128


def get_wisdm_settings():
    return WISDM_Settings

def get_wheelchair_settings():
    return Wheelchair_Settings