# HEPQPR.Qallse

The HEPQPR.Qallse project encodes the HEP (ATLAS) pattern recognition problem into a QUBO and solves it using a D-Wave or other classical QUBO libraries (qbsolv, neal).

The algorithm acts as a _doublet classifier_: the input is a large collection of double of hits, the output is a subset of those doublets that are believed to form true track candidates.


## Overview

### Algorithm overview 

![algorithm overview](https://docs.google.com/drawings/d/e/2PACX-1vRDn4qd1oqK9l03tova0ZH5UPl7-2OiH2kQ6h7YvTtGNiQafulbYC6XXa-EC_u4NITW-njDiLG4lEQ_/pub?w=960&amp;h=520)

### Current models

Different versions of the model building (i.e. QUBO generation) exist. They  are organised into a class hierarchy starting at the abstract class `hepqpr.qallse.QallseBase`:

* `.qallse.Qallse`: basic implementation, using constant bias weights. 
* `.qallse_mp.QallseMp`: adds a filtering step during triplets generation, which greatly limits the size of the QUBO;
* `.qallse_d0.QallseD0`: adds variable bias weights in the QUBO, based on the impact parameters `d0` and `z0`.


### Benchmarks

The datasets used for benchmarks can be recreated by executing the `scripts/recreate_datasets.py` script. All you need is to have the TrackML training set ("train_100_events.zip"). 

All benchmarks have initially been made using `QallseMp` (initial model). `QallseD0` is an improvement to limit the number of fakes, but relies on skewed physics assumptions. It has also been benchmarked (less thoroughly).

The raw results of all benchmarks are available here: [http://bit.ly/hepqpr-benchmark-stats](http://bit.ly/hepqpr-benchmark-stats).

### Performance overview

Timing: 

* model building is kind of slow: expect up to 1 hour for the biggest benchmark dataset;
* QUBO solving using qbsolv can be slow, especially using a D-Wave: expect up to 30 minutes in simulation (unbounded using a D-Wave, up to 5 hours in our experience)
* QUBO solving using neal is nearly instantaneous: up to 14 seconds;

Physics:

![Physics performance overview](https://docs.google.com/drawings/d/e/2PACX-1vTxS1sL5iPBzlmrpyVLOjENGDsnl7SZXzG-XIWHcpGl_WU-qfIsbJKOnNN0LqqstglHQAwPJpz_lJZP/pub?w=960&amp;h=400)

## Setup and usage

### Installation

Clone this repo, create a virtualenv and run `setup.py`. 

```bash
# clone
git clone <this repo>
cd <dir>

# create and activate virtualenv
python3 -m venv my_virtualenv
source ./my_virtualenv/bin/activate

# install
cd src
python setup.py install # or python setup.py develop for development
```

### Quickstart

```bash
# create a small dataset of 1% of a full event (be verbose)
# the -d option will also generate the input doublets
create_dataset -n 0.01 -p mini1 -d -v

# run the algorithm
qallse -i mini1/event000001000-hits.csv quickstart
```

### Commandline tools

The main commandline scripts are:

* `create_dataset`: to create datasets from TrackML events.
* `qallse`: run the algorithm.

Other tools are:

* `run_seeding`: generate the initial doublets, you won't need it if you call `create_dataset` with the `-d` option.
* `parse_qbsolv`: this parses a qbsolv logfile (with verbosity>=3) and generates a plot showing the energy of the solution after each main loop.
* `filter_doublets`: this can be used to remove doublets with too many holes from the input doublets.

Each tool comes with a `-h` or `--help` option.

### API

The `examples` directory contains some examples on how to do everything from scripts instead of using the commandline.

Other very useful functions are available in `hepqpr.qallse.cli.func` and pretty self-explanatory.


### Running from an IPython notebook 

Just create a conda environment and to install the package using `setup.py` (see [conda doc](https://conda.io/docs/user-guide/tasks/manage-environments.html)).

To get the output of qallse in the notebook, use:

```python
import logging
# turn on logging
logging.basicConfig()
# optional: display the time as well
fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", datefmt='%H:%M:%S')
for handler in logging.getLogger().handlers: handler.setFormatter(fmt)
# set the logging level for qallse
logging.getLogger('hepqpr').setLevel(logging.DEBUG)
```

See the notebook example for more information.

### Plotting

You can use `hepqpr.qallse.plotting` for plotting doublets and tracks easily. 

__Jupyter__: if you are running in a notebook, you need to tell the module so by calling `set_notebook_mode()`.

The methods take a `DataWrapper` and a list of xplets (an xplet is here a list of hit ids). The argument `dims` lets you define the plane to use (2D or 3D). The default is `xy`. 

Typical usage:

```python
from hepqpr.qallse.plotting import *
set_notebook_mode() # if running inside a notebook

# get the set of missing doublets
precision, recall, missings = dw.compute_score(final_doublets)

# plotting examples
iplot_results(dw, final_doublets, missings)
iplot_results(dw, final_doublets, missings, dims=list('zr'))
iplot_result_tracks(dw, final_tracks)
```


## Further information

### Datasets 

This code is intended to work on chunks of events from the [TrackML dataset](https://sites.google.com/site/trackmlparticle), which is representative of real HEP experiments.

When creating chunks/new datasets, the following simplifications are performed:

1. remove all hits from the endcaps;
2. keep only one instance of _double hits_ (i.e. duplicate signals on one volume from the same particle).

Then, a given percentage of particles and noisy hits a selected to be included in the new dataset. The `weight` used to compute the TrackML score are potentially modified ignore low pt and/or short particles (default cuts: <1 GeV, <5 hits) ==> __focused__ and __unfocused__ particles.

A command line script, `create_dataset` is available.
See the file `src/hepqpr/qallse/dsmaker/dsmaker.py` for more details.

### Metrics

The metrics used through the project are the TrackML score (computed using the [trackml-library](https://github.com/LAL/trackml-library)), the precision and the recall (see [wiki](https://en.wikipedia.org/wiki/Precision_and_recall)).

In this project:

* _precision_ = (number of focused doublets found) / (number of doublets in the solution - number of unfocused doublets found)
* _recall_ = (number of focused doublets found) / (number of true focused doublets)

### Vocabulary

Just as in the TrackML dataset documentation, we try to stick to _particles_ for the truth and to _tracks_ or _tracks candidates_ for reconstructed particle trajectories.

Doublets classifiers:

* _true_: when directly coming from the truth file;
* _real_: when corresponding to a true doublet. A real doublet can be:
    + _focused_ or 
    + _unfocused_
    depending on whether it belongs to a focused particle (depends on the pt and length cuts applied during dataset creation) or not;
* _fake_: a doublet that is not real;
* _missing_: a true doublet missing from the solution.


## About 

This code was produced during my Msc Thesis at the Lawrence Berkeley National Lab (LBNL). The subject:

> This MSc thesis studies the applicability of a special instance of quantum computing, adiabatic quantum optimization, as a potential source of superlinear speedup for particle tracking. The goal is to develop a track finding algorithm that runs on a D-Wave machine and to discuss the interest of Quantum Annealing as opposed to more conventional approaches.

Thank you to all the team that made it possible.
