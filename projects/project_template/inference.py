# %%
from january_ml.utils import load_config
import utils
import sys

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry

config = load_config("project_template")

# %%
x,y = utils.get_data(config["N_SAMPLES_INFER"], config["N_FEATURES"])

# %%
session = Session.builder.getOrCreate()
reg = Registry(session=session)

# %%
v = sys.argv[1]
print("HELLO THE VERSION IS", v)
model = reg.get_model("TESTMODEL").default

# %%
pred = model.run(x,function_name='predict')

# %%
pred

# %%



