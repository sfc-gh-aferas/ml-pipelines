from january_ml.utils import load_config
import utils

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry

config = load_config("example_project")

data = utils.get_data(config["N_SAMPLES_INFER"], config["N_FEATURES"])
X = data.drop(columns=["y"])

session = Session.builder.getOrCreate()
reg = Registry(session=session)

model_name = config["MODEL_NAME"]
model = reg.get_model(model_name).default

sdf = session.create_dataframe(X)
pred = model.run(sdf,function_name='predict')

pred.write.save_as_table(config["PRED_TABLE"],mode="overwrite")



