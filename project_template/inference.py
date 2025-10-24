# %%
import utils
import config as c

from snowflake.snowpark.session import Session
from snowflake.ml.registry import Registry

# %%
x,y = utils.get_data(c.N_SAMPLES_INFER)

# %%
session = Session.builder.getOrCreate()
reg = Registry(session=session)

# %%
model = reg.get_model("TESTMODEL").default

# %%
pred = model.run(x,function_name='predict')

# %%
pred

# %%



