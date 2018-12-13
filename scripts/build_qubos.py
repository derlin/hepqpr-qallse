import logging
# initialise the logging module
import pickle
import sys
import time

from hepqpr.qallse import *

# -------------------------------------------------

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d [%(name)-15s %(levelname)-5s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S')
logging.getLogger('hepqpr').setLevel(logging.DEBUG)

# -------------------------------------------------

# == INPUT CONFIG
densities = [10, 20, 30, 40, 50, 60, 70]
base_path = '/global/homes/l/llinder/current/hpt-collapse/ds'
event = 1004

# == RUN CONFIG

model_class = QallseMp  # model class to use
extra_config = dict(
    qubo_bias_weight=10,
    qubo_conflict_strength=20,
)

# -------------------------------------------------

csv_rows = ['ds,nts,nhits,ndblets,ip,ir,ims,drop,' +
            'q,qvars,qincl,qexcl,en0,cpu_time,wall_time']


def build_model(ds):
    path = f'{base_path}{ds}/event00000{event}'
    print(f'## path {path}')
    # load dataset
    dw = DataWrapper.from_path(path)
    # load doublets
    doublets = pd.read_csv(path + '-doublets.csv')

    # base scores
    num_tracks = dw.truth.particle_id.nunique() - 1  # -1 for noise
    ip, ir, ims = dw.compute_score(doublets)
    print(f'INPUT -- precision (%): {ip * 100:.4f}, recall (%): {ir * 100:.4f}, missing: {len(ims)}')

    wall_time = time.perf_counter()
    cpu_time = time.process_time()

    # instantiate qallse
    model = model_class(dw, **extra_config)
    # build the qubo
    model.build_model(doublets=doublets)
    Q, (qvars, qincl, qexcl) = model.to_qubo(return_stats=True)
    drops = model.get_build_stats().shape[0]
    en0 = dw.compute_energy(Q)

    cpu_time = time.process_time() - cpu_time
    wall_time = time.perf_counter() - wall_time

    with open(path + '-qubo.pickle', 'wb') as f:
        pickle.dump(Q, f)

    csv_rows.append(
        f'{ds},{num_tracks},{len(dw.hits)},{len(doublets)},{ip},{ir},{len(ims)},{drops},' +
        f'{len(Q)},{qvars},{qincl},{qexcl},{en0},{cpu_time},{wall_time}'
    )



# -------------------------------------------------

# run the experiment
for ds in densities:
    print(f'\n========> Running with density {ds}')
    build_model(ds)

# dump stats
stats = pd_read_csv_array(csv_rows, index_col=0)
stats.to_csv(f'build_qubos_stats-event{event}.csv')
print(stats.round(4).to_csv(sep='\t'))
#print(stats.round(4))
