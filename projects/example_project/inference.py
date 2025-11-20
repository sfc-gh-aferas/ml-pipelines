from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore, CreationMode
from january_ml.constants import FEATURE_SCHEMA


def main(session: Session):
    fs = FeatureStore(
        session=session,
        database=session.get_current_database(),
        name=FEATURE_SCHEMA,
        default_warehouse=session.get_current_warehouse(),
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST
    )

    fv = fs.get_feature_view("EXAMPLE_FEATURES",version="590DC9DFB32D518D5AD2CA1CC5537D6D")
    df = fs.read_feature_view(fv).sample(n=100)
    X = df.select("HOUR")

    reg = Registry(session=session)

    model_name = "MODEL_EX1"
    model = reg.get_model(model_name).default

    pred = model.run(X,function_name='predict')

    pred.write.save_as_table("TEST_MODEL_PREDICTIONS",mode="overwrite")



