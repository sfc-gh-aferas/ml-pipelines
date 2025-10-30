# %%
from january_ml.utils import load_config
from utils import get_data

from sklearn.linear_model import LinearRegression

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry

config = load_config('project_template')


# %%
x,y = get_data(config["N_SAMPLES_TRAIN"], config["N_FEATURES"])

# %%
lr = LinearRegression()
lr.fit(x,y)

# %%
session = Session.builder.getOrCreate()

# %%
reg = Registry(session=session)

# %%
reg.log_model(model=lr, model_name="TESTMODEL", sample_input_data=x)

# %%
reg.show_models()

# %%



