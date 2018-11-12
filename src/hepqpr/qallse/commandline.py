import logging

import click
import pandas as pd
import time
from datetime import timedelta

from .track_recreater import TrackRecreaterD
from .data_wrapper import DataWrapper


def _to_camelcase(text):
    """
    Converts underscore_delimited_text to CamelCase.
    Example: "tool_name" becomes "ToolName"
    """
    return ''.join(word.title() for word in text.split('_'))


DEFAULT_HITS_PATH = '/Users/lin/git/quantum-annealing-project/trackmlin/data/hpt_100/event000001000-hits.csv'


@click.command()
@click.option('--add-missing', is_flag=True, default=False)
@click.option('-c', '--cls', default='.qallse')
@click.option('-p', '--plot', is_flag=True, default=False)
@click.option('-e', '--extra', type=str, multiple=True)
@click.option('-i', '--input-path', default=DEFAULT_HITS_PATH)
def run(add_missing, cls, plot, extra, input_path):
    start_time = time.clock()

    # configure logging
    import sys
    logging.basicConfig(
        stream=sys.stderr,
        format="%(asctime)s [%(name)-15s %(levelname)-5s] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
    logging.getLogger('hepqpr').setLevel(logging.DEBUG)

    # load input data
    path = input_path.replace('-hits.csv', '')
    doublets = pd.read_csv(path + '-doublets.csv')
    dataw = DataWrapper.from_path(path)

    # parse any extra argument to pass to stimedge's constructor
    extra_config = dict()
    for s in extra:
        try:
            k, v = s.split('=')
            extra_config[k.strip()] = v.strip()
        except:
            print(f'error: {s} could not be processed. Extra args should be in the form k=v')

    # actually instantiate stimedge
    try:
        import importlib
        if cls.startswith('.'):
            module = __name__.replace('.commandline', '') + cls
            cls = cls[1:]
        else:
            module = '.'.join(cls.split('.')[:-1])
            cls = cls.split('.')[-1]
        ModelClass = getattr(importlib.import_module(module), _to_camelcase(cls))
        model = ModelClass(dataw, **extra_config)

    except Exception as err:
        raise RuntimeError(f'Error instantiating {cls}') from err

    # prepare doublets
    if add_missing:
        print('Cheat on, adding missing doublets.')
        doublets = dataw.add_missing_doublets(doublets)
    else:
        p, r, ms = dataw.compute_score(doublets)
        print(f'INPUT -- precision (%): {p * 100:.4f}%, recall (%): {r * 100:.4f}, missing: {len(ms)}')

    # build the qubo
    model.build_model(doublets=doublets)
    Q = model.to_qubo()
    # execute the qubo
    response = model.sample_qubo(Q=Q)
    # parse and postprocess the results
    output_doublets = model.process_sample(next(response.samples()))
    final_tracks, final_doublets = TrackRecreaterD().process_results(output_doublets)

    # stats
    en0 = dataw.compute_energy(Q)
    en = response.record.energy[0]
    print(f'SAMPLE -- energy: {en:.4f}, ideal: {en0:.4f} (diff: {en-en0:.6f})')
    occs = response.record.num_occurrences
    print(f'          best sample occurrence: {occs[0]}/{occs.sum()}')

    p, r, ms = dataw.compute_score(final_doublets)
    print(f'SCORE  -- precision (%): {p * 100}, recall (%): {r * 100}, missing: {len(ms)}')
    trackml_score = dataw.compute_trackml_score(final_tracks)
    print(f'          tracks found: {len(final_tracks)}, trackml score (%): {trackml_score * 100}')

    if plot:
        from .plotting import iplot_results, iplot_results_tracks
        iplot_results(dataw, final_doublets, ms, title='Best solution found (doublets)')
        iplot_results_tracks(dataw, final_tracks, dims=list('zxy'), title='Best solution found (tracks)', width=1500, height=1000)

    print(f'done in {timedelta(seconds=time.clock()-start_time)}')
