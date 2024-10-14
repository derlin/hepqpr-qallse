"""
This script recreates the datasets used in the benchmarks.

To run this script:

    - download the train_100_events.zip dataset from the TrackML challenge
    - update the BUILD CONFIG options below (input paths and output paths)

"""

import os.path as op

from qallse.cli.func import *
from qallse.dsmaker import create_dataset

# ==== BUILD CONFIG

loglevel = logging.DEBUG
trackml_train_path = "~/git/quantum-annealing-project/trackml-data/train_100_events/"

output_path = "/tmp/hpt-collapse"  # f'~/current/hpt-collapse

# ==== seeds used

ds_info = """
1000,0.1,1543636853
1000,0.2,1543636858
1000,0.3,1543636871
1000,0.4,1543636897
1000,0.5,1543636938
1000,0.6,1543637005
1000,0.7,1543637104
1004,0.1,1544857310
1004,0.2,1544857317
1004,0.3,1544857331
1004,0.4,1544857359
1004,0.5,1544857402
1004,0.6,1544857468
1004,0.7,1544857562
1054,0.1,1543703252
1054,0.2,1543703256
1054,0.3,1543703265
1054,0.4,1543703282
1054,0.5,1543703311
1054,0.6,1543703353
1054,0.7,1543703415
1039,1,1545352344
1062,1,1545352499
"""

# ==== configure logging

init_logging(logging.DEBUG, sys.stdout)

# ==== generation

headers = "event,percent,num_hits,num_noise,num_tracks,num_important_tracks,random_seed,cpu_time,wall_time".split(
    ","
)

if __name__ == "__main__":
    mat = []
    for row in ds_info.strip().split("\n"):
        e, d, s = row.split(",")
        event, ds, seed = int(e), float(d), int(s)
        prefix = f"ds{ds*100:.0f}"

        print(f"\n>>>> {prefix} <<<<\n")
        with time_this() as time_info:
            metas, path = create_dataset(
                density=ds,
                input_path=op.join(trackml_train_path, f"event00000{event}-hits.csv"),
                output_path=output_path,
                prefix=prefix,
                min_hits_per_track=5,
                high_pt_cut=1.0,
                random_seed=int(seed),
                double_hits_ok=False,
                gen_doublets=True,
            )

        mat.append(
            [
                event,
                int(ds * 100),
                metas["num_hits"],
                metas["num_noise"],
                metas["num_tracks"],
                metas["num_important_tracks"],
                seed,
                time_info[0],
                time_info[1],
            ]
        )

    stats = pd.DataFrame(mat, columns=headers)
    stats.to_csv("recreate_datasets.csv", index=False)
