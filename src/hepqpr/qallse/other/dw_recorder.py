"""
Utility for recording of timing QPU execution when using QBSolv.
See https://gist.githubusercontent.com/dexter2206/916e407e4ce88475ea93e20d2516f78f/raw/4d74767a431486c9c3ad0c6dacb4fd6f9eb006bf/qpu_timing_example.py

## Usage

Instantiate a DWave solver:

>>> from dwave_qbsolv import QBSolv
>>> from dwave.system.samplers import DWaveSampler
>>> from dwave.system.composites import EmbeddingComposite
>>> sampler = EmbeddingComposite(DWaveSampler(config_file='/path/to/dwave.conf', permissive_ssl=True))

Run qbsolv, using our recorder:
>>> with record_sampler_invocations(sampler) as records:
>>>   QBSolv().sample_qubo(Q, solver=sampler)

In this example we take only sample_qubo calls, but sample_ising calls are recorded as well:
>>> qubo_timings = [record['timing'] for record in records if record['method'] == 'qubo']
>>> total_sampling_time = sum(record['qpu_sampling_time'] for record in qubo_timings)
>>> print('Average QPU sampling time [QUBO]: {}'.format(total_sampling_time / len(qubo_timings)))
"""
from contextlib import contextmanager
from functools import wraps
import time


def _make_recorder(func, target, key, store_arguments=False):
    """Wrap given callable in wrapper that records returned timings and used arguments."""

    @wraps(func)
    def _wrapped(*args, **kwargs):
        record = {}
        if store_arguments:
            record['args'] = args
            record['kwargs'] = kwargs
        wall_time = time.perf_counter()
        result = func(*args, **kwargs)
        record['wall_time'] = (time.perf_counter() - wall_time)
        record['timing'] = result.info['timing']
        record['overhead'] = record['wall_time'] - (record['timing']['total_real_time'] * 1E-6)
        record['method'] = key
        target.append(record)
        return result

    return _wrapped


@contextmanager
def record_sampler_invocations(sampler, store_arguments=False):
    """Turn on temporary recording of sampler invocations."""
    overriden_methods = ['qubo', 'ising']
    target = []
    wrappers, originals = [], dict()
    try:
        # wrap the sample methods, the timing information will be saved in the
        # target array
        for method in overriden_methods:
            func = getattr(sampler, f'sample_{method}')
            originals[method] = func
            setattr(sampler, f'sample_{method}', _make_recorder(func, target, method, store_arguments))

        # return a reference to the target array
        yield target

    finally:
        # restore the "normal" methods
        for method in overriden_methods:
            setattr(sampler, f'sample_{method}', originals[method])