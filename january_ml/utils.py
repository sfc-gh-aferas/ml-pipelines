import yaml
import os

def load_config(project_name: str) -> dict:
    if os.path.basename(os.getcwd()) == project_name:
        config_dir = ""
    elif project_name in os.listdir(os.getcwd()):
        config_dir = f"{project_name}/"
    elif "projects" in os.listdir(os.getcwd()):
        config_dir = f"projects/{project_name}/"
    else:
        raise ValueError("Project name not findable from current working directory:", os.getcwd())
    return yaml.safe_load(open(config_dir+"config.yml",'r'))