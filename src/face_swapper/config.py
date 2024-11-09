from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def get_checkpoints_dir() -> Path:
    return Path(Path(get_project_root(), 'data'),'checkpoints')

def get_images_dir() -> Path:
    return Path(Path(get_project_root(), 'data'),'images')
