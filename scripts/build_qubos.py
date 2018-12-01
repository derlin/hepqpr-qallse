# import modules
import logging
# initialise the logging module
import pickle
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
tmpdir_path = '/tmp/hpt-collapse'

# == DATASET CREATION CONFIG

random_seed = 17932241798878

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
    qubo_bias_weight=1,
    qubo_conflict_strength=2,
)
qbsolv_options = dict(
    timeout=3 * 60 * 60,  # in seconds => 3 hrs
    seed=random_seed
)

# -------------------------------------------------

csv_rows = ['idx,nts,nhits,ndblets,ip,ir,ims,drop,' +
            'q,qvars,qincl,qexcl,en0,time']


def build_model(n, dw, doublets, path):
    num_tracks = dw.truth.particle_id.nunique() - 1  # -1 for noise
    ip, ir, ims = dw.compute_score(doublets)

    exec_time = time.process_time()

    # instantiate qallse
    model = model_class(dw, **extra_config)
    # build the qubo
    model.build_model(doublets=doublets)
    Q, (qvars, qincl, qexcl) = model.to_qubo(return_stats=True)
    drops = model.get_build_stats().shape[0]
    en0 = dw.compute_energy(Q)
    exec_time = time.process_time() - exec_time

    csv_rows.append(
        f'{n},{num_tracks},{len(dw.hits)},{len(doublets)},{ip},{ir},{len(ims)},{drops},' +
        f'{len(Q)},{qvars},{qincl},{qexcl},{en0},{exec_time}'
    )

    doublets.to_csv(path + '-doublets.csv', index=False)
    with open(path + '-qubo.pickle', 'wb') as f:
        pickle.dump(Q, f)

    return model


def generate_data_and_run(tmpdir, n):
    ds_options[param_name] = n
    metadata, path = create_dataset(
        path=input_path, output_path=tmpdir, **ds_options)
    print(f'dataset generated in {path}')
    # load dataset
    dw = DataWrapper.from_path(path)
    # generate doublets
    doublets = generate_doublets(path + '-hits.csv')
    if add_missing: doublets = dw.add_missing_doublets(doublets)
    p, r, ms = dw.compute_score(doublets)
    print(f'INPUT -- precision (%): {p * 100:.4f}, recall (%): {r * 100:.4f}, missing: {len(ms)}')
    # run model
    return build_model(n, dw, doublets, path)


# -------------------------------------------------

# generate tmp directory
import tempfile

tmpdir = None
if tmpdir_path is None:
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = tmpdir.name

# run the experiment
for n in param_values:
    print(f'\n========> Running with {n}')
    generate_data_and_run(tmpdir_path, n)

# dump stats
stats = pd_read_csv_array(csv_rows, index_col=0)
stats.to_csv('stats.csv')
print(stats.round(4).to_csv(sep='\t'))
#print(stats.round(4))