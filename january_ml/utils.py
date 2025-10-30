import yaml
import os

def load_config(project_name: str) -> dict:
    if "config.yml" in os.listdir(os.getcwd()):
        config_dir = ""
    elif project_name in os.listdir(os.getcwd()):
        config_dir = f"{project_name}/"
    elif "projects" in os.listdir(os.getcwd()):
        config_dir = f"projects/{project_name}/"
    else:
        raise ValueError("Config file not findable from current working directory:", os.getcwd())
    
    config = yaml.safe_load(open(config_dir+"config.yml",'r'))
    if config["project_name"] == project_name:
        return config
    raise ValueError(f"Config file at {config_dir} is for project {config['project_name']}, not {project_name}")