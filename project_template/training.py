# %%
import utils
import config as c
from sklearn.linear_model import LinearRegression

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry

# %%
x,y = utils.get_data(c.N_SAMPLES_TRAIN)

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



