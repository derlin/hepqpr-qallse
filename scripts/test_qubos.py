# import modules
import logging
# initialise the logging module
import pickle
import sys
import time

from dwave_qbsolv import QBSolv
from hepqpr.qallse import *

# -------------------------------------------------

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')

logging.getLogger('hepqpr').setLevel(logging.DEBUG)

# -------------------------------------------------

# == EXPERIMENT CONFIG
bias_weights = [-.1, 0, .1]
conflict_strengths = [.5, .8, 1.]

# == DATASET CONFIG

base_path = '/global/homes/l/llinder/hpt-collapse/ez-%s_hpt-1/event000001000'
# base_path = '/tmp/hpt-collapse/ez-%s_hpt-1/event000001000'
ds_sizes = [.1, .2, .3, .4, .5, .6]
min_hits_per_track = 5
num_repeats = 3

# == RUN CONFIG

random_seed = 17932241798878
qbsolv_options = dict(
    timeout=3 * 60 * 60,  # in seconds => 3 hrs
    seed=random_seed
)

# -------------------------------------------------

csv_rows = ['percent,bias_weight,conflict_strength,i,'
            'qtime,en,en0,endiff,occ,sum_occ,'
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


def process_sample(sample):
    kept_triplets = [Triplet.name_to_hit_ids(k) for k, v in sample.items() if v == 1]
    all_doublets: List = np.unique(tracks_to_xplets(kept_triplets), axis=0).tolist()
    print(f'KEPT -- triplets={len(kept_triplets)}, doublets={len(all_doublets)}')
    # recreate tracks and resolve remaining conflicts
    tr = TrackRecreaterD()
    f_tracks, f_doublets = tr.process_results(all_doublets)
    final_tracks = [t for t in f_tracks if len(t) >= min_hits_per_track]
    final_doublets = tracks_to_xplets(final_tracks, x=2)

    return final_doublets, final_tracks, len(tr.conflicts)

def run_experiment(percent):
    path = base_path % str(percent)

    dw = DataWrapper.from_path(path)
    doublets = pd.read_csv(path + '-doublets.csv')
    with open(path + '-qubo.pickle', 'rb') as f:
        Qo = pickle.load(f)

    p, r, ms = dw.compute_score(doublets)
    print(f'INPUT -- precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')

    for bweight in bias_weights:
        for cstrength in conflict_strengths:
            print(f'========> running bias_weight={bweight}, conflict_strength={cstrength}')
            Q = transform_qubo(Qo, bweight, cstrength)
            en0 = dw.compute_energy(Q)

            for i in range(num_repeats):
                exec_time = time.process_time()
                # execute the qubo
                qtime = time.process_time()
                response = QBSolv().sample_qubo(Q=Q, **qbsolv_options)
                qtime = time.process_time() - qtime

                # get all output doublets
                sample = next(response.samples())
                final_doublets, final_tracks, cs = process_sample(sample)
                exec_time = time.process_time() - exec_time
                print(f'       final_tracks={len(final_tracks)}, final_doublets={len(final_doublets)}')

                # stats about the qbsolv run
                en = response.record.energy[0]
                occs = response.record.num_occurrences
                s = ""
                for e,o in zip(response.record.energy, occs):
                    s += f',{e}|{o}'
                print('en|occ: ', s[1:])
                print(f'SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})')
                print(f'          best sample occurrence: {occs[0]}/{occs.sum()}')

                # scores
                n_valid, n_invalid, _ = dw.get_score_numbers(final_doublets)
                p, r, ms = dw.compute_score(final_doublets)
                print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}')
                trackml_score = dw.compute_trackml_score(final_tracks)
                print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')

                row = \
                    f'{percent},{bweight},{cstrength},{i},' \
                    f'{qtime},{en},{en0},{en-en0},{occs[0]},{occs.sum()},' \
                    f'{len(dw.get_real_doublets())},{len(final_doublets)},{n_valid},{n_invalid},' \
                    f'{p},{r},{trackml_score},{len(ms)},{cs},{exec_time}'

                print('CSV', row)
                csv_rows.append(row)

# -------------------------------------------------

# run the experiment
for percent in ds_sizes:
    print(f'\n\n**************************** Running with {percent}')
    run_experiment(percent)

# dump stats
stats = pd_read_csv_array(csv_rows, index_col=0)
stats.to_csv('stats.csv')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(stats.round(4))
