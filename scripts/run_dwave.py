# import modules
import logging
import pickle
import random
import time

from dwave_qbsolv import QBSolv
from hepqpr.qallse import *
from hepqpr.qallse.other.stdout_redirect import *

# import D-Wave modules
#from dwave.system.samplers import DWaveSampler
#from dwave.system.composites import EmbeddingComposite

# -------------------------------------------------

# == EXPERIMENT CONFIG

bias_weight = -0.1
conflict_strength = 1.0
num_repeats = 4

# == DATASET CONFIG

# path = '/global/homes/l/llinder/hpt-collapse/ez-0.6_hpt-1/event000001000'
path = '/tmp/hpt-collapse/ez-0.1_hpt-1/event000001000'
min_hits_per_track = 5

# == RUN CONFIG

# random_seed = 17932241798878
qbsolv_options = dict(
    timeout=3 * 60 * 60,
    num_reads=10,
    num_repeats=10,
    verbosity=4,
)

# path to the dwave.conf configuration file (created using dwave-config create)
dwave_conf = '/Users/lin/git/quantum-annealing-project/dwave-leap/conf/dwave.conf'

# -------------------------------------------------

# initialise the logging module
logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger('hepqpr').setLevel(logging.DEBUG)


# -------------------------------------------------

def transform_qubo(Q, bias_weight, conflict_strength):
    Q2 = dict()
    for k, v in Q.items():
        if v == 1:
            Q2[k] = bias_weight
        elif v == 2:
            Q2[k] = conflict_strength
        else:
            Q2[k] = v
    return Q2


def run_dwave():
    # D-Wave solver used by qbsolv for the sampling of sub-QUBOs
    # sampler = DWaveSampler(config_file=dwave_conf)
    # solver = EmbeddingComposite(sampler)
    solver = None  # TODO
    random_seed = random.randint(0, 1 << 30)
    print(f'using random seed: {random_seed}')

    # Prepare data
    dw = DataWrapper.from_path(path)
    doublets = pd.read_csv(path + '-doublets.csv')
    with open(path + '-qubo.pickle', 'rb') as f:
        Qo = pickle.load(f)

    Q = transform_qubo(Qo, bias_weight, conflict_strength)
    p, r, ms = dw.compute_score(doublets)
    print(f'INPUT -- {len(doublets)} precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')

    exec_time = time.process_time()
    # run qbsolv
    with stdout_redirect(f'qbsolv-{random_seed}.log'):
        response = QBSolv().sample_qubo(Q=Q, solver=solver, seed=random_seed, **qbsolv_options)
    qtime = time.process_time() - exec_time

    # get all output doublets
    sample = next(response.samples())
    kept_triplets = [Triplet.name_to_hit_ids(k) for k, v in sample.items() if v == 1]
    all_doublets: List = np.unique(tracks_to_xplets(kept_triplets), axis=0).tolist()

    # recreate tracks and resolve remaining conflicts
    tr = TrackRecreaterD()
    final_tracks, final_doublets = tr.process_results(all_doublets, min_hits_per_track=min_hits_per_track)
    exec_time = time.process_time() - exec_time
    print(f'       final_tracks={len(final_tracks)}, final_doublets={len(final_doublets)}')

    # dump all to file
    with open(f'qbsolv_response.pickle', 'wb') as f:
        pickle.dump(response, f)
    with open(f'final_tracks_doublets.pickle', 'wb') as f:
        pickle.dump((final_tracks, final_doublets), f)

    print('CSV seed,qtime,time')
    print(f'CSV {random_seed},{qtime},{exec_time}')


run_dwave()
