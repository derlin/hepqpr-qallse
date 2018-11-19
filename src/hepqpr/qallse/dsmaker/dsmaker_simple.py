#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import re
import tempfile
from datetime import datetime

import click
import pandas as pd
import numpy as np
import json

BARREL_VOLUME_IDS = [8, 13, 17]

import logging

logger = logging.getLogger(__name__)


def create_dataset(
        path, output_path,
        percent=.1, min_hits_per_track=3,
        high_pt_cut=.0, double_hits_ok=False,
        prefix=None, random_seed=None):
    # capture all parameters, so we can dump them to a file later
    input_params = locals()

    # initialise random
    if random_seed is None:
        random_seed = int(datetime.now().timestamp())
    random.seed(random_seed)

    event_id = re.search('(event[0-9]+)', path)[0]

    # for computing track density in the end
    phi_angle = 2 * np.pi
    theta_angle = np.pi

    # compute the prefix
    if prefix is None:
        prefix = f'ez-{percent}'
        if high_pt_cut > 0:
            prefix += f'_hpt-{high_pt_cut}'
        else:
            prefix += '_baby'
        if double_hits_ok:
            prefix += '_dbl'

    # ---------- prepare data

    # load the data
    hits = pd.read_csv(path + '-hits.csv')
    particles = pd.read_csv(path + '-particles.csv')
    truth = pd.read_csv(path + '-truth.csv')

    # add indexes
    particles.set_index('particle_id', drop=False, inplace=True)
    truth.set_index('hit_id', drop=False, inplace=True)
    hits.set_index('hit_id', drop=False, inplace=True)

    # create a merged dataset with hits and truth
    df = hits.join(truth, rsuffix='_', how='inner')

    logger.debug(f'Loaded {len(df)} hits from {path}.')

    # ---------- filter hits

    # keep only hits in the barrel region
    df = df[hits.volume_id.isin(BARREL_VOLUME_IDS)]
    logger.debug(f'Filtered hits from barrel. Remaining hits: {len(df)}.')

    # store the noise for later, then remove them from the main dataframe
    # do this before filtering double hits, as noise will be thrown away as duplicates
    noise_df = df.loc[df.particle_id == 0]
    df = df[df.particle_id != 0]

    if not double_hits_ok:
        df.drop_duplicates(['particle_id', 'volume_id', 'layer_id'], keep='first', inplace=True)
        logger.debug(f'Dropped double hits. Remaining hits: {len(df)}.')

    # ---------- sample tracks

    num_tracks = int(df.particle_id.nunique() * percent)
    sampled_particle_ids = random.sample(df.particle_id.values.tolist(), num_tracks)
    df = df[df.particle_id.isin(sampled_particle_ids)]

    # ---------- sample noise

    num_noise = int(len(noise_df) * percent)
    sampled_noise = random.sample(noise_df.hit_id.values.tolist(), num_noise)
    noise_df = noise_df.loc[sampled_noise]

    # ---------- recreate hits, particle and truth

    new_hit_ids = df.hit_id.values.tolist() + noise_df.hit_id.values.tolist()
    new_hits = hits.loc[new_hit_ids]
    new_truth = truth.loc[new_hit_ids]
    new_particles = particles.loc[sampled_particle_ids]

    # ---------- fix truth

    if high_pt_cut > 0:
        # set low pt weights to 0
        hpt_mask = np.sqrt(truth.tpx ** 2 + truth.tpy ** 2) >= high_pt_cut
        new_truth.loc[~hpt_mask, 'weight'] = 0
        logger.debug(f'High Pt hits: {sum(hpt_mask)}/{len(new_truth)}')

    if min_hits_per_track > 0:
        short_tracks = new_truth.groupby('particle_id').filter(lambda g: len(g) < min_hits_per_track)
        new_truth.loc[short_tracks.index, 'weight'] = 0

    # ---------- write data

    # write the dataset to disk
    output_path = os.path.join(output_path, prefix)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, event_id)

    new_hits.to_csv(output_path + '-hits.csv', index=False)
    new_truth.to_csv(output_path + '-truth.csv', index=False)
    new_particles.to_csv(output_path + '-particles.csv', index=False)

    # ---------- write metadata

    density_denom = 2 * np.pi ** 2
    track_density = percent / density_denom
    print(f'Dataset track density: {track_density}')

    metadata = dict(
        num_hits=new_hits.shape[0],
        num_tracks=num_tracks,
        num_important_tracks=new_truth[new_truth.weight != 0].particle_id.nunique(),
        num_noise=num_noise,
        track_density=num_tracks / density_denom,
        hit_density=len(new_hits) / density_denom,
        random_seed=random_seed,
        time=datetime.now().isoformat(),
    )
    for k, v in metadata.items():
        logger.debug(f'  {k}={v}')

    metadata['params'] = input_params

    with open(output_path + '-meta.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    return metadata, output_path


DEFAULT_INPUT_PATH = '/Users/lin/git/quantum-annealing-project/trackml-data/train_100_events/event000001000'
DEFAULT_OUTPUT_PATH = '/tmp'


def generate_tmp_datasets(n=10, input_path=DEFAULT_INPUT_PATH, *ds_args, **ds_kwargs):
    for _ in range(n):
        tp = tempfile.TemporaryDirectory()
        yield create_dataset(input_path, tp.name, *ds_args, **ds_kwargs)


@click.command()
@click.option('-n', '--density', type=click.FloatRange(0, 1), default=.1,
              help='The sampling to apply, in percent.')
@click.option('--hpt', type=float, default=0,
              help='Only select tracks with a transverse momentum '
                   'higher or equal than FLOAT (in GeV, inclusive)')
@click.option('--double-hits/--no-double-hits', is_flag=True, default=False,
              help='Keep only one instance of double hits.')
@click.option('-h', '--min-hits', type=int, default=5,
              help='The minimum number of hits per tracks (inclusive)')
@click.option('-p', '--prefix', type=str, default=None,
              help='Prefix for the dataset output directory')
@click.option('-s', '--seed', type=str, default=None,
              help='Seed to use when initializing the random module')
@click.option('-d', '--doublets', is_flag=True, default=False,
              help='Generate doublets as well')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Be verbose.')
@click.option('-o', '--output-path', default=DEFAULT_OUTPUT_PATH,
              help='Where to create the dataset directoy')  # tempfile.gettempdir())
@click.option('-i', 'input_path', default=DEFAULT_INPUT_PATH,
              help='Path to the original event hits file')
def cli(density, hpt, double_hits, min_hits, prefix, seed,
        doublets, verbose, output_path, input_path):
    if verbose:
        import sys
        logging.basicConfig(
            stream=sys.stderr,
            format="%(asctime)s [dsmaker] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S',
            level=logging.DEBUG)

    meta, path = create_dataset(
        input_path, output_path,
        density, min_hits,
        hpt, double_hits,
        prefix, seed)

    seed, density = meta['random_seed'], meta['track_density']
    print(f'Dataset written in {path}* (seed={seed}, track density={density})')

    if doublets:
        from hepqpr.qallse.seeding import generate_doublets
        doublets_df = generate_doublets(path + '-hits.csv')
        with open(os.path.join(path + '-doublets.csv'), 'w') as f:
            doublets_df.to_csv(f, index=False)
            print(f'Doublets (len={len(doublets_df)}) generated.')


if __name__ == "__main__":
    cli()
