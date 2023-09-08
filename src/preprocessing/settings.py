from pathlib import Path

class BaseSettings:
    # Local Settings
    # base_path: Path = Path("/Users/wkong0407/Documents/yairi-lab/parker-preprocessing-comparison")
    # raw_path: Path = base_path / "raw_data"
    # ABCI Settings
    base_path: Path = Path("/home/acf15412ed/parker-preprocessing-comparison")
    raw_path: Path = base_path / "raw_data"
class WISDM_Settings(BaseSettings):

    segment_path: Path = BaseSettings.raw_path / "WISDM_segmented" / "segments"
    label_path: Path = BaseSettings.raw_path / "WISDM_segmented" / "labels"

    processed_path: Path = BaseSettings.base_path / "processed_data" / "wisdm"
    yaml_path: Path = BaseSettings.base_path / "wisdm.yaml"

class Wheelchair_Settings(BaseSettings):

    segment_path: Path = BaseSettings.raw_path / "車椅子_segmented" / "segments"
    label_path: Path = BaseSettings.raw_path / "車椅子_segmented" / "labels"

    processed_path: Path = BaseSettings.base_path / "processed_data" / "wheelchair"
    yaml_path: Path = BaseSettings.base_path / "車椅子.yaml"

def get_wisdm_settings():
    return WISDM_Settings

def get_wheelchair_settings():
    return Wheelchair_Settings