import click
import re

SOL_NEW_BEST = 'new best energy'
SOL_NEW = 'new energy'
SOL_DUPLICATE_BEST = 'best energy (dup)'
SOL_DUPLICATE = 'duplicate energy'
SOL_NOTHING = ''

sol_types = dict(
    NOTHING=0,  # nothing new, do nothing
    NEW_HIGH_ENERGY_UNIQUE_SOL=1,  # solution is unique, highest new energy
    DUPLICATE_HIGHEST_ENERGY=2,  # two cases, solution is unique, duplicate energy
    DUPLICATE_ENERGY=3,  # two cases, solution is duplicate highest energy
    DUPLICATE_ENERGY_UNIQUE_SOL=4,  # two cases, solution is unique, highest energy
    NEW_ENERGY_UNIQUE_SOL=5  # solution is unique, new highest energy
)
colors = ['black', 'red', 'darkred', 'lightblue', 'blue', 'yellow']


class Iterator:

    def __init__(self, items):
        self.len = len(items)
        self.cursor = 0
        self.iter = iter(items)

    def next(self):
        if self.has_next():
            self.cursor += 1
            return next(self.iter)
        return None

    def has_next(self):
        return self.cursor < self.len

    def __len__(self):
        return self.len


_extract_time = lambda line: float(re.search('^\d+\.\d*', line)[0])
_extract_energy = lambda line: float(re.match('.*((loop =)|(answer  ))(-?\d+\.\d*)', line).groups()[-1])


def parse(lines):
    started = False
    lines = Iterator(lines)

    best_energy = None
    answers = []
    times = []
    annotations = []

    while lines.has_next():
        line = lines.next()

        if not started:
            if 'Energy of solution' in line:
                started = True
                while 'Starting outer loop' not in line:
                    line = lines.next()  # skip the state dump
                times.append(_extract_time(line))
                answers.append(_extract_energy(line))
                annotations.append('NEW_HIGH_ENERGY_UNIQUE_SOL')

        elif 'after partition pass' in line:
            while not 'Latest answer' in line:
                line = lines.next()
            times.append(_extract_time(line))
            answers.append(_extract_energy(line))
            annotations.append(lines.next().strip().split(' ')[0])
            while not 'V Best outer loop' in line:
                line = lines.next()
            best_energy = _extract_energy(line)

    return times, answers, best_energy, annotations


def plot_energies(times, answers, annotations):
    from plotly.offline import plot
    import plotly.graph_objs as go
    import numpy as np

    best = min(answers)
    best_idx = answers.index(best)
    traces = [go.Scatter(
        x=times,
        y=answers,
        mode='lines+markers',
        name='solutions over time',
        text=annotations,
        hoverinfo='x+y+text',
        marker=dict(color=[colors[sol_types[a]] for a in annotations])
    ), go.Scatter(
        x=[times[best_idx], times[-1]],
        y=[best] * 2,
        name='best',
        opacity=.5
    )]
    plot(traces, filename='temp-qbsolv.html')


@click.command()
@click.option('-i', '--input', type=str)
@click.option('--plot/--no-plot', is_flag=True, default=True)
def cli(input, plot):
    with open(input) as f:
        times, answers, best_energy, annotations = parse(f.readlines())
        print(','.join(map(str, answers)))
        print(','.join(map(str, times)))
        print('Best: ', best_energy)

        if plot:
            plot_energies(times, answers, annotations)
    print('done')


if __name__ == "__main__":
    cli()
