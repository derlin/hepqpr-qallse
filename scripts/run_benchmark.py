# import modules
import logging
# initialise the logging module
import sys
import time

from hepqpr.qallse import *
from hepqpr.qallse.dsmaker.dsmaker_simple import create_dataset
from hepqpr.qallse.seeding import generate_doublets

# -------------------------------------------------

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger('hepqpr').setLevel(logging.DEBUG)

# -------------------------------------------------

# == EXPERIMENT CONFIG
param_name = 'percent'
param_values = [.01, .02]  # [.10, .20, .30, .40, .50, .60]

# == DATASET CREATION CONFIG

random_seed = 42

event_id = 1000
input_path = f'/Users/lin/git/quantum-annealing-project/trackml-data/train_100_events/event00000{event_id}'
#input_path = f'/global/homes/l/llinder/train_100_events/event00000{event_id}'

ds_options = dict(
    min_hits_per_track=5,
    high_pt_cut=1,
    double_hits_ok=False,
    random_seed=random_seed,
)

# == INPUT CONFIG

# whether or not to add missing doublets to the input
add_missing = False

# == RUN CONFIG

model_class = QallseMp  # model class to use
extra_config = dict(
    qubo_conflict_strength=0.8,
    qubo_bias_weight=.0,
)
qbsolv_options = dict(
    timeout=3 * 60 * 60,  # in seconds => 3 hrs
    seed=17932241798878
)

# -------------------------------------------------

csv_rows = ['idx,nts,nhits,ndblets,ip,ir,ims,drop,' +
            'q,qvars,qincl,qexcl,qtime,en,en0,endiff,' +
            'real,gen,valid,invalid,' +
            'p,r,trackml,ms,cs,time']


def run_model(n, dw, doublets):
    num_tracks = dw.truth.particle_id.nunique() - 1  # -1 for noise
    ip, ir, ims = dw.compute_score(doublets)

    exec_time = time.process_time()

    # instantiate qallse
    model = model_class(dw, **extra_config)
    # build the qubo
    model.build_model(doublets=doublets)
    Q, (qvars, qincl, qexcl) = model.to_qubo(return_stats=True)

    # execute the qubo
    qtime = time.process_time()
    response = model.sample_qubo(Q=Q, **qbsolv_options)
    qtime = time.process_time() - qtime

    # get all output doublets
    all_doublets = model.process_sample(next(response.samples()))
    # recreate tracks and resolve remaining conflicts
    tr = TrackRecreaterD()
    f_tracks, f_doublets = tr.process_results(all_doublets)
    final_tracks = [t for t in f_tracks if len(t) >= ds_options['min_hits_per_track']]
    final_doublets = tracks_to_xplets(final_tracks, x=2)
    exec_time = time.process_time() - exec_time

    # stats about the qbsolv run
    drops = len(model.get_build_stats())
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

    csv_rows.append(
        f'{n},{num_tracks},{len(dw.hits)},{len(doublets)},{ip},{ir},{len(ims)},{drops},' +
        f'{len(Q)},{qvars},{qincl},{qexcl},{qtime},{en},{en0},{en-en0},' +
        f'{len(dw.get_real_doublets())},{len(final_doublets)},{n_valid},{n_invalid},' +
        f'{p},{r},{trackml_score},{len(ms)},{len(tr.conflicts)},{exec_time}'
    )

    model.input_doublets = doublets
    model.Q = Q
    model.ms = ms
    model.response = response
    model.final_doublets = final_doublets
    model.final_tracks = final_tracks

    return model


def generate_data_and_run(tmpdir, n):
    ds_options[param_name] = n
    metadata, path = create_dataset(
        path=input_path, output_path=tmpdir.name, **ds_options)
    print(f'dataset generated in {path}')
    # load dataset
    dw = DataWrapper.from_path(path)
    # generate doublets
    doublets = generate_doublets(path + '-hits.csv')
    if add_missing: doublets = dw.add_missing_doublets(doublets)
    p, r, ms = dw.compute_score(doublets)
    print(f'INPUT -- precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')
    # run model
    return run_model(n, dw, doublets)


# -------------------------------------------------

# generate tmp directory
import tempfile

tmpdir = tempfile.TemporaryDirectory()

# run the experiment
for n in param_values:
    print(f'\n========> Running with {n}')
    generate_data_and_run(tmpdir, n)

# dump stats
stats = pd_read_csv_array(csv_rows, index_col=0)
stats.to_csv('stats.csv')
#print(stats.to_csv(sep='\t'))
print(stats.round(4))