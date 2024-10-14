"""
This example show how to sample a QUBO using qbsolv in simulation.

To run this example:

    - run the `build_qubo.py` script using NO placeholder (i.e. if you change the options in the script,
        you will have to update this scripts as well)

"""

raise NotImplementedError(
    "Dwave's QBsolv is deprecated as of 2022. "
    "The hybrid Dwave solver requires refactoring."
)

import sys
import logging
import pickle
from os.path import join as path_join

from qallse import *
from qallse.other.stdout_redirect import capture_stdout

from dwave_qbsolv import QBSolv

# ==== RUN CONFIG


loglevel = logging.DEBUG

input_path = "/tmp/ez-0.1_hpt-1.0/event000001000-hits.csv"  # TODO change it !
qubo_path = "/tmp"  # TODO change it
logfile = "/tmp/qbsolv.log"  # try to run parse_qbsolv -i /tmp/qbsolv.log afterwards :)

sampler = QBSolv()
sampler_args = dict(
    timeout=3 * 60 * 60,
    num_repeats=10,
    verbosity=4,
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
with open(path_join(qubo_path, "qubo.pickle"), "rb") as f:
    Q = pickle.load(f)

# sample qubo
with capture_stdout(logfile):
    response = sampler.sample_qubo(Q, **sampler_args)

# get the results
all_doublets = Qallse.process_sample(next(response.samples()))
final_tracks, final_doublets = TrackRecreaterD().process_results(all_doublets)

# compute stats
en0 = dw.compute_energy(Q)
en = response.record.energy[0]
occs = response.record.num_occurrences

p, r, ms = dw.compute_score(final_doublets)
trackml_score = dw.compute_trackml_score(final_tracks)

# print stats
print(f"SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})")
print(f"          best sample occurrence: {occs[0]}/{occs.sum()}")

print(f"SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}")
print(
    f"          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}"
)
