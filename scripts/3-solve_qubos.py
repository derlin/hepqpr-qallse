"""
This script is a template/inspiration that you can use to solve qubos for benchmark purpose.
It will output a csv file with the statistics called solve_qubos_{solver}.csv in the current directory.

To run this script:

- create benchmark datasets: the convention is a directory per density (-n option of create_dataset),
  round to an integer and with the ds prefix: density 0.1 in folder ds10, etc.
- build qubos using for example the script 2-build_qubos.py
- update the values in BUILD CONFIG

"""

import pickle
import random

from qallse.cli.func import *
import os.path as op

# ==== RUN CONFIG TODO change it

loglevel = logging.DEBUG

events = [1000]
dss = [10]
repeat = 1

data_path = (
    "/tmp/hpt-collapse/ds{ds}/event00000{event}-hits.csv"  # path to the datasets
)

qubo_path = "/tmp"  # path where the qubos are pickled
qubo_prefix = "evt{event}-ds{ds}-"  # prefix for the qubo files
output_path = "/tmp/"  # where to serialize the responses
output_prefix = qubo_prefix  # prefix for serialized responses

solver = "neal"  # solver to use
solver_config = (
    dict()
)  # parameters for the solver. Note that "seed" is generated later.

# ==== configure logging

init_logging(logging.DEBUG, sys.stdout)


# ==== build model


def run_one(event, ds):
    # load data
    path = data_path.format(event=event, ds=ds)
    dw = DataWrapper.from_path(path)
    qubo_filepath = op.join(
        qubo_path, qubo_prefix.format(event=event, ds=ds) + "qubo.pickle"
    )

    with open(qubo_filepath, "rb") as f:
        Q = pickle.load(f)
    en0 = dw.compute_energy(Q)

    for i in range(repeat):
        # set seed
        seed = random.randint(0, 1 << 30)
        solver_config["seed"] = seed

        # build model
        with time_this() as time_info:
            with time_this() as qtime_info:
                if solver == "neal":
                    response = solve_neal(Q, **solver_config)
                elif solver == "qbsolv":
                    response = solve_qbsolv(Q, **solver_config)
                elif solver == "dwave":
                    response = solve_dwave(Q, **solver_config)
                else:
                    raise Exception("Invalid solver name.")

            final_doublets, final_tracks = process_response(response)

        # compute scores
        p, r, ms = dw.compute_score(final_doublets)
        trackml = dw.compute_trackml_score(final_tracks)
        en = response.record.energy[0]

        # output composition
        _, _, d_real = diff_rows(dw.get_real_doublets(), final_doublets)
        _, d_fakes, d_real_all = diff_rows(
            dw.get_real_doublets(with_unfocused=True), final_doublets
        )

        # save response
        output_filename = op.join(
            output_path,
            output_prefix.format(event=event, ds=ds) + f"{solver}-{i}-response.pickle",
        )

        with open(output_filename, "wb") as f:
            pickle.dump(response, f)

        # gather stats
        mat.append(
            [
                event,
                ds,
                seed,
                i,
                len(final_tracks),
                len(dw.get_real_doublets()),
                len(final_doublets),
                len(d_real),
                len(d_real_all),
                len(d_fakes),
                p,
                r,
                trackml,
                len(ms),
                en,
                en0,
                en - en0,
            ]
            + qtime_info
            + time_info
        )


if __name__ == "__main__":
    mat = []

    for event in events:
        for ds in dss:
            print(f"\n==========>>> processing event {event} ds{ds}\n", flush=True)
            run_one(event, ds)

    headers = (
        "event,percent,seed,repeat,tracks_gen,"
        "n_true,n_gen,n_real,n_real_all,n_fakes,"
        "precision,recall,trackml,missings,"
        "en,en0,endiff,"
        "qtime_cpu,qtime_wall,cpu_time,wall_time"
    )

    stats = pd.DataFrame(mat, columns=headers.split(","))
    stats.to_csv(f"solve_qubos_{solver}.csv", index=False)
