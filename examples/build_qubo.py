"""
This example show how to save a model for later benchmarks. The output will be two files:

* `qubo.pickle`: the QUBO in the form of a dictionary
* `xplets.json`: a dictionary with the xplets used in the QUBO

To run this example:

    - generate a dataset using `create_dataset` from the command line
    - update the input_path below

"""

import sys
import logging

from qallse import *
from qallse import dumper

# ==== BUILD CONFIG

loglevel = logging.DEBUG

input_path = "/tmp/ez-0.1_hpt-1.0/event000001000-hits.csv"  # TODO change it !
output_path = "/tmp"  # TODO change it

model_class = QallseD0  # model class to use
extra_config = dict()  # model config

dump_config = dict(
    output_path="/tmp",
    prefix="",
    xplets_kwargs=dict(
        format="json", indent=3
    ),  # use json (vs "pickle") and indent the output
    qubo_kwargs=dict(
        w_marker=None, c_marker=None
    ),  # save the real coefficients VS generic placeholders
)

# ==== configure logging

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logging.getLogger("hepqpr").setLevel(loglevel)

# ==== build model

# load data
dw = DataWrapper.from_path(input_path)
doublets = pd.read_csv(input_path.replace("-hits.csv", "-doublets.csv"))

# build model
model = model_class(dw, **extra_config)
model.build_model(doublets)

# dump model to a file
dumper.dump_model(model, **dump_config)
