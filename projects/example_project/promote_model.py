from january_ml.utils import load_config

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry


config = load_config('example_project')

def main(session: Session, model_version: str) -> None:

    # Get model
    reg = Registry(session=session)

    model_name = config["MODEL_NAME"]

    base_model = reg.get_model(model_name)
    mv = base_model.version(model_version)
    base_model.default = mv