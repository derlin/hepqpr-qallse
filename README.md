
```text
                $$$$$$$                 
            $$$$$$$$$$$$$$              
         $$$$$$$$$$$$$$$$$$             
        $$$$$$$$$$$$$$$$$$$$            
       $$$$$$$$$$$$$$$☻$$$$$    $$$$$$  
       $$$$$$$$☻$$$$$$$$$$$$   $$$  $$$ 
       $$$$$$$$$$$$$$$$$$$$  $$$$    $$ 
        $$$$$$$$$$$$$$$$$$$$$$$$        
         $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
 $$$$$$     $$$$$$$$$$$$$$$$$$        $$
$$   $$$$$$$$$$$$$$$$$$$$$$$         $$ 
 $$$    $$$$$$$$$$$$$$$$$$$$$$$$$$$$    
     $$$$$$$$  $$$ $$$$$$          $$   ________           .__   .__                   
   $$$$       $$$  $$$ $$$      $$$$    \_____  \  _____   |  |  |  |    ______  ____  
  $$$       $$$$   $$$  $$$              /  / \  \ \__  \  |  |  |  |   /  ___/_/ __ \ 
   $$$$$   $$$     $$$   $$$$    $$     /   \_/.  \ / __ \_|  |__|  |__ \___ \ \  ___/ 
    $$$    $$$$$$  $$$    $$$$$$$$      \_____\ \_/(____  /|____/|____//____  > \___  >
            $$$     $$$$$   $$$                \__>     \/                  \/      \/ 
```

# qallse

The [HEPQPR](https://sites.google.com/lbl.gov/hep-qpr).Qallse project encodes the HEP (ATLAS) pattern recognition problem into a QUBO and solves it using a D-Wave or other classical QUBO libraries (qbsolv, neal).

The algorithm acts as a _doublet classifier_: the input is a large collection of potential doublets, the output is a subset of those doublets that are believed to form true track candidates.

## Contribution and reuse

This code is under an Apache 2.0 license, so you are free to do pretty much everything you want with it ;).
 
However, I put a lot of work and time on it, so it is easy to use/read/fork/understand. If you happen to be interested, I would really appreciate if you could **add a star to the project** and use the Github fork mechanism (and mention the repo/the author in case you present your results somewhere). 

I am available for any question (email or Github issue is fine) and would be glad to hear about your ideas and improvements !  🐙🐙


## Content

  * [Overview](#overview)
    + [Algorithm overview](#algorithm-overview)
    + [Current models](#current-models)
    + [Benchmarks](#benchmarks)
    + [Performance overview](#performance-overview)
  * [Setup and usage](#setup-and-usage)
    + [Installation](#installation)
    + [Quickstart](#quickstart)
    + [Commandline tools](#commandline-tools)
    + [Solving QUBOs with qbsolv and D-Wave](#solving-qubos-with-qbsolv-and-d-wave)
    + [API](#api)
    + [Running from an IPython notebook](#running-from-an-ipython-notebook)
  * [Plotting](#plotting)
    + [The plotting module](#the-plotting-module)
    + [Exporting plots as pdf (or other formats)](#exporting-plots-as-pdf)
  * [Further information](#further-information)
    + [Datasets](#datasets)
    + [Metrics](#metrics)
    + [Vocabulary](#vocabulary)
  * [Resources](#resources)
  * [About](#about)

## Overview

### Algorithm overview 

![algorithm overview](https://docs.google.com/drawings/d/e/2PACX-1vRDn4qd1oqK9l03tova0ZH5UPl7-2OiH2kQ6h7YvTtGNiQafulbYC6XXa-EC_u4NITW-njDiLG4lEQ_/pub?w=960&amp;h=520)

### Current models

Different versions of the model building (i.e. QUBO generation) exist. They  are organised into a class hierarchy starting at the abstract class `qallse.QallseBase`:

* `.qallse.Qallse`: basic implementation, using constant bias weights. 
* `.qallse_mp.QallseMp`: adds a filtering step during triplets generation, which greatly limits the size of the QUBO;
* `.qallse_d0.QallseD0`: adds variable bias weights in the QUBO, based on the impact parameters `d0` and `z0`.


### Benchmarks

The datasets used for benchmarks can be recreated by executing the `scripts/recreate_datasets.py` script. All you need is to have the TrackML training set ("train_100_events.zip"). 

All benchmarks have initially been made using `QallseMp` (initial model). `QallseD0` is an improvement to limit the number of fakes, but relies on skewed physics assumptions. It has also been benchmarked (less thoroughly).

The raw results of all benchmarks are available here: [http://bit.ly/hepqpr-benchmark-stats](http://bit.ly/hepqpr-benchmark-stats).

### Performance overview

__Timing__:

* model building is kind of slow: expect up to 1 hour for the biggest benchmark dataset;
* QUBO solving using qbsolv can be slow, especially using a D-Wave: expect up to 30 minutes in simulation (unbounded using a D-Wave, up to 5 hours in our experience)
* QUBO solving using neal is nearly instantaneous: up to 14 seconds.

__Physics__

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
create_dataset -n 0.01 -p mini1 -v

# run the algorithm
qallse -i mini1/event000001000-hits.csv quickstart
```

### Commandline tools

The main commandline scripts are:

* `create_dataset`: to create datasets from TrackML events.
* `qallse`: run the algorithm.

Other tools are:

* `run_seeding`: generate the initial doublets, you only need it if you call `create_dataset` with the `--no-doublets` option.
* `parse_qbsolv`: this parses a qbsolv logfile (with verbosity>=3) and generates a plot showing the energy of the solution after each main loop.
* `filter_doublets`: this can be used to remove doublets with too many holes from the input doublets.

Each tool comes with a `-h` or `--help` option.

__Typical example__:

```bash
> mkdir /tmp/mini

# generate a dataset of 5% in /tmp/mini/ds05
> create_dataset -o /tmp/mini -p ds05 -n 0.05
Dataset written in /tmp/mini/ds05/event000001000* (seed=376778465, num. tracks=409)

# build the model
> qallse -i /tmp/mini/ds05/event000001000-hits.csv -o /tmp/mini build
INPUT -- precision (%): 0.8610, recall (%): 99.5885, missing: 1
2019-01-29T09:54:05.691 [qallse.qallse_d0 INFO ] created 15341 doublets.
2019-01-29T09:54:06.995 [qallse.qallse_d0 INFO ] created 3160 triplets.
2019-01-29T09:54:07.022 [qallse.qallse_d0 INFO ] created 686 quadruplets.
2019-01-29T09:54:07.022 [qallse.qallse_d0 INFO ] Model built in 3.12s. doublets: 15341/0, triplets: 3160/0, quadruplets: 686
2019-01-29T09:54:07.030 [qallse.qallse_d0 INFO ] MaxPath done in 0.02s. doublets: 544, triplets: 628, quadruplets: 638 (dropped 48)
2019-01-29T09:54:07.073 [qallse.qallse_d0 INFO ] Qubo generated in 0.07s. Size: 2877. Vars: 628, excl. couplers: 1611, incl. couplers: 638
Wrote qubo to /tmp/mini/qubo.pickle

# solve using neal
> qallse -i /tmp/mini/ds05/event000001000-hits.csv -o /tmp/mini neal
2019-01-29T09:56:51.207 [qallse.cli.func INFO ] QUBO of size 2877 sampled in 0.14s (NEAL, seed=1615186406).
2019-01-29T09:56:51.619 [qallse.track_recreater INFO ] Found 0 conflicting doublets
SAMPLE -- energy: -165.7110, ideal: -163.1879 (diff: -2.523028)
          best sample occurrence: 1/10
SCORE  -- precision (%): 99.1769547325103, recall (%): 99.1769547325103, missing: 2
          tracks found: 48, trackml score (%): 99.38064159235999
Wrote response to /tmp/mini/neal_response.pickle

# plot the results
qallse -i /tmp/mini/ds05/event000001000-hits.csv -o /tmp/mini plot -r /tmp/mini/neal_response.pickle
```

### Solving QUBOs with qbsolv and D-Wave

__qbsolv logs__: the `qallse qbsolv` commandline tool is quite rich. Here is an example on how to visualise the main loops of qbsolv (using the qubo created in the previous section):

```bash
# !!!!!!!! ensure there no buffered io !!!!!!!!
export PYTHONUNBUFFERED=1

# get the qbsolv logs into a file (verbosity should be at least 3)
qallse -i /tmp/mini/ds05/event000001000-hits.csv -o /tmp/mini qbsolv \
    -l /tmp/qbsolv.log \
    -v 4
    
# plot the energies after each main loop
parse_qbsolv -i /tmp/qbsolv.log
```

__D-Wave__: the only thing you need is a valid [D-Wave configuration file](https://docs.ocean.dwavesys.com/en/latest/overview/dwavesys.html#configuring-a-d-wave-system-as-a-solver)
(you can create an account on the [D-Wave LEAP cloud platform](https://cloud.dwavesys.com/leap/) to get 1 minute of QPU time for free).
Then, simply use the `-dw` option and that's it ! The sub-QUBOs are now solved on a D-Wave:

```bash
qallse -i /tmp/mini/ds05/event000001000-hits.csv -o /tmp/mini qbsolv \
    -dw /path/to/dwave.conf
```

### API

The `examples` directory contains some examples on how to do everything from scripts instead of using the commandline.

Other very useful functions are available in `qallse.cli.func` and pretty self-explanatory.


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

## Plotting

### The plotting module
 
You can use `qallse.plotting` for plotting doublets and tracks easily. 

__Jupyter__: if you are running in a notebook, you need to tell the module so by calling `set_notebook_mode()`.

The methods take a `DataWrapper` and a list of xplets (an xplet is here a list of hit ids). The argument `dims` lets you define the plane to use (2D or 3D). The default is `xy`. A lot more options are available, just look at the source.

Typical usage:

```python
from qallse import DataWrapper
from qallse.cli.func import process_response
from qallse.plotting import *

set_notebook_mode() # if running inside a notebook

# load the dataset and the response (created using the qallse tool)
dw = DataWrapper.from_path('/path/to/eventx-hits.csv')
with open('/path/to/response.pickle', 'rb') as f: 
    import pickle
    response = pickle.load(f)

# process the response and get the set of missing doublets
final_doublets, final_tracks = process_response(response)
precision, recall, missings = dw.compute_score(final_doublets)

# plotting examples
iplot_results(dw, final_doublets, missings)
iplot_results(dw, final_doublets, missings, dims=list('zr'))
iplot_results_tracks(dw, final_tracks)
```
### Exporting plots as pdf

Use the `return_fig` argument to get hold of the `Figure` object, then use the `orca` tool as described in the [plotly documentation](https://plot.ly/python/static-image-export/). Note that extra arguments are passed to the plotly layout object constructor. 
Here is a complicated example:

```python
import plotly.io as pio

fig = iplot_results(
    dw, final_doublets, missings, 
    yaxis=dict(range=[0, 1100]), # clip the Y axis
    xaxis=dict(range=[-500, 500]), # clip the X axis
    width=600, height=700, # change the figure size
    legend=dict(font=dict(size=16)), # bigger font in legend
    show_buttons=False, # don't show the interactive buttons
    shapes=xy_layer_shapes, # draw the layers
    return_fig=True # return the figure object
)

pio.write_image(fig, '/tmp/foo.pdf')
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


## Resources

**general** 

* [http://bit.ly/hepqpr-benchmark-stats](http://bit.ly/hepqpr-benchmark-stats) Google Sheet with all the statistics concerning my experiments
* Master Thesis report --> see the `doc` folder (online and print versions available)
* [http://bit.ly/hepqpr-qallse](http://bit.ly/hepqpr-qallse) Slides presenting the command-line tools and APIs in this repo + Some interesting getting started slides for the D-Wave APIs (as of January, 2019)
* [https://arxiv.org/abs/1902.08324](https://arxiv.org/abs/1902.08324) Publication "*A pattern recognition algorithm for quantum annealers*"

**Talks**

* [http://bit.ly/hepqpr-cpad](http://bit.ly/hepqpr-cpad) Talk given at [*CPAD'18*](http://www.brown.edu/Conference/CPAD2018/) (Providence, USA, December 2018): general overview (20')
* [http://bit.ly/hepqpr-cdots](http://bit.ly/hepqpr-cdots) Talk given at [*Connecting the Dots 2019*](https://indico.cern.ch/event/742793/) (Valencia, Spain, April 2019): general overview (20')
* [http://bit.ly/hepqpr-apr](http://bit.ly/hepqpr-apr) Talk given at [*Learning to Discover – Advanced pattern Recognition*](https://ipa-user.universite-paris-saclay.fr/learning-to-discover) (Paris, France, November 2019): more in-depth presentation (45')


## About 

This code was produced during my Msc Thesis at the Lawrence Berkeley National Lab (LBNL). The subject:

> This MSc thesis studies the applicability of a special instance of quantum computing, adiabatic quantum optimization, as a potential source of superlinear speedup for particle tracking. The goal is to develop a track finding algorithm that runs on a D-Wave machine and to discuss the interest of Quantum Annealing as opposed to more conventional approaches.

Thank you to all the team that made it possible.
