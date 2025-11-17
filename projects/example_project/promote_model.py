from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry


def main(session: Session, model_version: str) -> None:

    # Get model
    reg = Registry(session=session)

    model_name = "TEST_MODEL"

    base_model = reg.get_model(model_name)
    mv = base_model.version(model_version)
    base_model.default = mv