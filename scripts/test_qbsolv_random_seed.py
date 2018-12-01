# import modules
import logging
# initialise the logging module
import pickle
import random
import sys
import time

from dwave_qbsolv import QBSolv
from hepqpr.qallse import *

# -------------------------------------------------

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger('hepqpr').setLevel(logging.DEBUG)

# -------------------------------------------------

# == EXPERIMENT CONFIG

bias_weight = 0.1
conflict_strength = .8

num_repeats = 3
num_seeds = 5

#logfile = 'qbsolv_ds03_%d.out'

# == DATASET CONFIG

path = 'hpt-collapse/ez-0.3_hpt-1/event000001000'
#path = '/tmp/hpt-collapse/ez-0.02_hpt-1/event000001000'
min_hits_per_track = 5

# == RUN CONFIG

# random_seed = 17932241798878
qbsolv_options = dict(
    timeout=3 * 60 * 60
)

# -------------------------------------------------

csv_rows = ['seed,repeat,qtime,en,en0,endiff,'
            'n_real,n_gen,n_valid,n_invalid,p,r,trackml,ms,cs,time']


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


def run_experiment():
    dw = DataWrapper.from_path(path)
    doublets = pd.read_csv(path + '-doublets.csv')
    with open(path + '-qubo.pickle', 'rb') as f:
        Qo = pickle.load(f)

    Q = transform_qubo(Qo, bias_weight, conflict_strength)
    p, r, ms = dw.compute_score(doublets)
    print(f'INPUT -- precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')

    for _ in range(num_seeds):
        random_seed = random.random()
        print(f'========> running with seed={random_seed}')

        for i in range(num_repeats):
            print(f'******* attempt {i} *******')
            exec_time = time.process_time()
            # run qbsolv
            qtime = time.process_time()
            #with stdout_redirect(logfile % random_seed):
            response = QBSolv().sample_qubo(Q=Q, **qbsolv_options)
            qtime = time.process_time() - qtime

            # get all output doublets
            sample = next(response.samples())
            kept_triplets = [Triplet.name_to_hit_ids(k) for k, v in sample.items() if v == 1]
            all_doublets_dups = tracks_to_xplets(kept_triplets)
            all_doublets: List = np.unique(all_doublets_dups, axis=0).tolist()
            print(f'DIFF -- kept triplets={len(kept_triplets)}, doublets={len(all_doublets)}/{len(all_doublets_dups)}')
            # recreate tracks and resolve remaining conflicts
            tr = TrackRecreaterD()
            f_tracks, f_doublets = tr.process_results(all_doublets)
            final_tracks = [t for t in f_tracks if len(t) >= min_hits_per_track]
            final_doublets = tracks_to_xplets(final_tracks, x=2)
            exec_time = time.process_time() - exec_time
            print(f'       final_tracks={len(final_tracks)}, final_doublets={len(final_doublets)}')

            # stats about the qbsolv run
            en0 = dw.compute_energy(Q)
            en = response.record.energy[0]
            print(f'SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})')
            occs = response.record.num_occurrences
            print(f'          best sample occurrence: {occs[0]}/{occs.sum()}')

            # scores
            n_valid, n_invalid, _ = dw.get_score_numbers(final_doublets)
            p, r, ms = dw.compute_score(final_doublets)
            print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}')
            trackml_score = dw.compute_trackml_score(final_tracks)
            print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')

            row = \
                f'{random_seed},{i},' \
                f'{qtime},{en},{en0},{en-en0},' \
                f'{len(dw.get_real_doublets())},{len(final_doublets)},{n_valid},{n_invalid},' \
                f'{p},{r},{trackml_score},{len(ms)},{len(tr.conflicts)},{exec_time}'
            print('CSV', row)
            csv_rows.append(row)


# -------------------------------------------------

run_experiment()

# dump stats
stats = pd_read_csv_array(csv_rows, index_col=0)
stats.to_csv('stats.csv')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(stats.round(4))
