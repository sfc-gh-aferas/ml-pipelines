# %%
import argparse

parser = argparse.ArgumentParser("test")

parser.add_argument(
    "--woo",
)
args = parser.parse_args()

__return__ = {args.woo:"beep"}
# %%



