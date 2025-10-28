import yaml

def load_config(config_path: str) -> dict:
    return yaml.safe_load(open(config_path,'r'))