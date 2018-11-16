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
        num_tracks=500, num_noise=0, min_hits_per_track=3,
        high_pt_cut=.0, barrel_only=True, double_hits_ok=False,
        phi_bounds=None, theta_bounds=None,
        prefix=None, random_seed=None):
    # initialise random
    if random_seed is None:
        random_seed = int(datetime.now().timestamp())
    random.seed(random_seed)

    # capture all parameters, so we can dump them to a file later
    all_params = locals()
    event_id = re.search('(event[0-9]+)', path)[0]

    # for computing track density in the end
    phi_angle = 2 * np.pi
    theta_angle = np.pi

    # compute the prefix
    if prefix is None:
        if high_pt_cut > 0:
            prefix = f'hpt-{high_pt_cut}'
        elif barrel_only:
            prefix = 'barrel'
        else:
            prefix = 'baby'

        if double_hits_ok:
            prefix += '_dbl'
        if phi_bounds is not None:
            prefix += f'_phi_{phi_bounds[0]}-{phi_bounds[1]}'

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

    if barrel_only:
        # keep only hits in the barrel region
        df = df[hits.volume_id.isin(BARREL_VOLUME_IDS)]
        logger.debug(f'Filtered hits from barrel. Remaining hits: {len(df)}.')

    if high_pt_cut > 0:
        # get only hits with a high pt
        df = df[np.sqrt(df.tpx ** 2 + df.tpy ** 2) > high_pt_cut]
        logger.debug(f'Filtered high PT tracks. Remaining hits: {len(df)}.')

    if phi_bounds is not None:
        df['phi'] = np.arctan2(df.y, df.x)
        df = df[(df.phi >= phi_bounds[0]) & (df.phi <= phi_bounds[1])]
        logger.debug(f'Filtered using phi bounds {phi_bounds}. Remaining hits: {len(df)}.')
        phi_angle = phi_bounds[1] - phi_bounds[0]

    if theta_bounds is not None:
        df['theta'] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), df.z)
        df = df[(df.theta >= theta_bounds[0]) & (df.theta <= theta_bounds[1])]
        logger.debug(f'Filtered using theta bounds {theta_bounds}. Remaining hits: {len(df)}.')
        theta_angle = theta_bounds[1] - theta_bounds[0]

    # store the noise for later, before dropping double hits, since the drop_duplicates
    # will remove all noise with the same volume and layer id ...
    noise_df = df.loc[df.particle_id == 0]

    if not double_hits_ok:
        df.drop_duplicates(['particle_id', 'volume_id', 'layer_id'], keep='first', inplace=True)
        logger.debug(f'Dropped double hits. Remaining hits: {len(df)}.')

    # ---------- recreate and sample tracks

    # filter tracks with not enough hits
    tracks, oops = [], []
    for particle_id, df_p in df[df.particle_id != 0].groupby('particle_id'):
        track = (particle_id, df_p.hit_id.values.tolist())
        if df_p.shape[0] >= min_hits_per_track:
            tracks.append(track)
        else:
            oops.append(track)

    logger.debug(f'Recreated {len(tracks)} tracks with at least {min_hits_per_track} hits.')

    # do a random choice
    if len(tracks) == 0:
        raise RuntimeError(f'ERROR: no track matching those arguments. Aborting')

    if num_tracks > 0:
        if len(tracks) < num_tracks:
            print(f'WARNING: not enough tracks. Selecting {len(tracks)} only.')
            final_tracks = tracks
            num_tracks = len(tracks)
        else:
            final_tracks = random.sample(tracks, num_tracks)
        assert len(final_tracks) <= num_tracks
    else:
        final_tracks = tracks
        num_tracks = len(final_tracks)

    new_particle_ids, hids = zip(*final_tracks)
    new_track_hids = sum(list(hids), [])
    final_particle_ids = list(new_particle_ids)
    final_hits_ids = list(new_track_hids)
    assert len(final_tracks) == len(final_particle_ids)

    logger.debug(f'Sampled {len(final_tracks)} tracks using {len(final_hits_ids)} hits.')

    # ---------- add noise

    # compute the number of noise to add
    nnoise = 0
    if num_noise == -1:
        nnoise = len(noise_df)  # add all
    elif 0 < num_noise < 1:
        # this is a percentage
        nnoise = int(num_noise * len(final_hits_ids))
    else:
        nnoise = num_noise

    # add noise, if any
    if nnoise > 0:
        if nnoise >= len(noise_df):
            final_hits_ids += noise_df.hit_id.values.tolist()
        else:
            final_hits_ids += noise_df.sample(nnoise).hit_id.values.tolist()
        logger.debug(f'Added {nnoise} noise hits.')

    # ---------- create final dataframes

    new_hits = hits.loc[final_hits_ids]
    new_truth = truth.loc[final_hits_ids]
    new_particles = particles.loc[final_particle_ids]

    # ---------- fix truth

    new_truth.loc[~new_truth.hit_id.isin(new_track_hids), 'weight'] = 0
    new_truth.weight = new_truth.weight / new_truth.weight.sum()

    # ---------- write data

    # write the dataset to disk
    output_path = os.path.join(output_path, f'{prefix}_{num_tracks}')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, event_id)

    new_hits.to_csv(output_path + '-hits.csv', index=False)
    new_truth.to_csv(output_path + '-truth.csv', index=False)
    new_particles.to_csv(output_path + '-particles.csv', index=False)

    track_density = num_tracks / (phi_angle * theta_angle)
    print(f'Dataset track density: {track_density}')

    metadata = dict(params=all_params, track_density=track_density)
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
@click.option('--barrel/--no-barrel', is_flag=True, default=True,
              help='Only select hits located in the barrel')
@click.option('--double-hits/--no-double-hits', is_flag=True, default=False,
              help='Keep only one instance of double hits.')
@click.option('--hpt', type=float, default=0,
              help='Only select tracks with a transverse momentum higher than FLOAT (in GeV, exclusive)')
@click.option('-t', '--num-tracks', type=int, default=100,
              help='The number of tracks to include')
@click.option('-h', '--min-hits', type=int, default=5,
              help='The minimum number of hits per tracks (inclusive)')
@click.option('-n', '--num-noise', type=float, default=0,
              help='The number of hits not part of any tracks to include. If < 1, it is interpreted as a percentage.')
@click.option('--phi-bounds', type=click.FloatRange(0, 2 * np.pi), default=None, nargs=2,
              help='Only select tracks located in the given phi interval (in [0, 2π] rad)')
@click.option('--theta-bounds', type=click.FloatRange(0, np.pi), default=None, nargs=2,
              help='Only select tracks located in the given theta interval (in [0, π] rad)')
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
def cli(barrel, hpt, double_hits, num_tracks, min_hits, num_noise, phi_bounds, theta_bounds, prefix, seed,
        doublets, verbose, output_path, input_path):
    if verbose:
        import sys
        logging.basicConfig(
            stream=sys.stderr,
            format="%(asctime)s [dsmaker] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S',
            level=logging.DEBUG)

    # click is passing an empty tuple by default...
    if len(phi_bounds) != 2: phi_bounds = None
    if len(theta_bounds) != 2: theta_bounds = None

    new_seed, path = create_dataset(
        input_path, output_path,
        num_tracks, num_noise, min_hits,
        hpt, barrel, double_hits,
        phi_bounds, theta_bounds,
        prefix, seed)

    print(f'Dataset written in {path}* (seed={new_seed})')

    if doublets:
        from hepqpr.qallse.seeding import generate_doublets
        doublets_df = generate_doublets(path + '-hits.csv')
        with open(os.path.join(path + '-doublets.csv'), 'w') as f:
            doublets_df.to_csv(f, index=False)
            print(f'Doublets (len={len(doublets_df)}) generated.')


if __name__ == "__main__":
    cli()
