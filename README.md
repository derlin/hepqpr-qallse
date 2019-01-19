# HEPQPR.Qallse

The HEPQPR.Qallse project encodes the HEP (ATLAS) pattern recognition problem into a QUBO and solves it using a D-Wave or other classical QUBO libraries (qbsolv, neal).

The algorithm acts as a _doublet classifier_: the input is a large collection of double of hits, the output is a subset of those doublets that are believed to form true track candidates.

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

# run the algorithm (add -p to see the results in your browser)
run_qallse -i mini1/event000001000-hits.csv 
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
