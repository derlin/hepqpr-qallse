# import modules
import logging
import pickle
import random
import time

from dwave_qbsolv import QBSolv
from hepqpr.qallse import *
from hepqpr.qallse.other.stdout_redirect import *

# -------------------------------------------------

# initialise the logging module
logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger('hepqpr').setLevel(logging.DEBUG)

# -------------------------------------------------

# == EXPERIMENT CONFIG

bias_weight = -0.1
conflict_strength = 1.0
num_repeats = 4

# == DATASET CONFIG


# path = '/global/homes/l/llinder/hpt-collapse/ez-0.6_hpt-1/event000001000'
# path = '/tmp/hpt-collapse/ez-0.02_hpt-1/event000001000'
min_hits_per_track = 5

# == RUN CONFIG

# random_seed = 17932241798878
qbsolv_options = dict(
    timeout=3 * 60 * 60,
    verbosity=4,
)

# -------------------------------------------------

csv_rows = ['seed,repeat,qtime_cpu,qtime_wall,en,en0,endiff,'
            'n_real,n_gen,n_valid,n_invalid,p,r,trackml,ms,cs,cpu_time,wall_time']


def transform_qubo(Q, bias_weight, conflict_strength):
    Q2 = dict()
    for k, v in Q.items():
        if v == 10:
            Q2[k] = bias_weight
        elif v == 20:
            Q2[k] = conflict_strength
        else:
            Q2[k] = v
    return Q2


def run_experiment(path, prefix=''):
    dw = DataWrapper.from_path(path)
    doublets = pd.read_csv(path + '-doublets.csv')
    with open(path + '-qubo.pickle', 'rb') as f:
        Qo = pickle.load(f)

    Q = transform_qubo(Qo, bias_weight, conflict_strength)
    p, r, ms = dw.compute_score(doublets)
    print(f'INPUT -- {len(doublets)}, precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')

    for i in range(num_repeats):
        random_seed = random.randint(0, 1 << 30)

        wall_time = time.perf_counter()
        cpu_time = time.process_time()

        # run qbsolv
        qtime_wall = time.perf_counter()
        qtime_cpu = time.process_time()
        with stdout_redirect(f'{prefix}qbsolv_{i}-{random_seed}.log'):
            response = QBSolv().sample_qubo(Q=Q, seed=random_seed, **qbsolv_options)
            time.sleep(1)
        qtime_cpu = time.process_time() - qtime_cpu
        qtime_wall = time.perf_counter() - qtime_wall

        # get all output doublets
        sample = next(response.samples())
        kept_triplets = [Triplet.name_to_hit_ids(k) for k, v in sample.items() if v == 1]
        print('kept triplets:', len(kept_triplets))
        all_doublets: List = np.unique(tracks_to_xplets(kept_triplets), axis=0).tolist()

        # recreate tracks and resolve remaining conflicts
        tr = TrackRecreaterD()
        final_tracks, final_doublets = tr.process_results(all_doublets, min_hits_per_track=min_hits_per_track)

        cpu_time = time.process_time() - cpu_time
        wall_time = time.perf_counter() - wall_time

        print(f'       final_tracks={len(final_tracks)}, final_doublets={len(final_doublets)}')

        # dump all to file
        with open(f'{prefix}qbsolv_response_{i}-{random_seed}.pickle', 'wb') as f: pickle.dump(response, f)
        with open(f'{prefix}final_tracks_doublets_{i}-{random_seed}.pickle', 'wb') as f: pickle.dump(
            (final_tracks, final_doublets), f)

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
            f'{qtime_cpu},{qtime_wall},{en},{en0},{en-en0},' \
            f'{len(dw.get_real_doublets())},{len(final_doublets)},{n_valid},{n_invalid},' \
            f'{p},{r},{trackml_score},{len(ms)},{len(tr.conflicts)},{cpu_time},{wall_time}'
        print('CSV', row)
        csv_rows.append(row)


# -------------------------------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        print('missing path to hit file')
        sys.exit(1)

    path = sys.argv[1].replace('-hits.csv', '')
    prefix = '' if len(sys.argv) == 2 else sys.argv[2]
    print(f'Running on dataset {path} (prefix={prefix})')
    run_experiment(path, prefix)

    # dump stats
    stats = pd_read_csv_array(csv_rows, index_col=0)
    stats.to_csv(f'{prefix}stats.csv')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(stats.round(4))
