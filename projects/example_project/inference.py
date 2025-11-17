import utils

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry


data = utils.get_data(100,5)
X = data.drop(columns=["y"])

session = Session.builder.getOrCreate()
reg = Registry(session=session)

model_name = "TEST_MODEL"
model = reg.get_model(model_name).default

sdf = session.create_dataframe(X)
pred = model.run(sdf,function_name='predict')

pred.write.save_as_table("TEST_MODEL_PREDICTIONS",mode="overwrite")



