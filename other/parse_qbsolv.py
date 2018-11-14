import click
import re


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


def parse(lines):
    started = False
    lines = Iterator(lines)

    best_energy = None
    answers = []
    times = []

    while lines.has_next():
        line = lines.next()

        if not started:
            if 'Energy of solution' in line:
                started = True
                while 'Starting outer loop' not in line:
                    line = lines.next()  # skip the state dump
                times.append(float(re.search('^\d+\.\d*', line)[0]))
                answers.append(float(re.match('.*loop =(-?\d+\.\d*)', line).group(1)))

        elif 'after partition pass' in line:
            while not 'Latest answer' in line:
                line = lines.next()
            times.append(float(re.search('^\d+\.\d*', line)[0]))
            answers.append(float(re.match('.*answer  (-?\d+\.\d*)', line).group(1)))
            while not 'V Best outer loop' in line:
                line = lines.next()
            best_energy = float(re.match('.*loop =(-?\d+\.\d*)', line).group(1))

    return times, answers, best_energy


def plot_energies(times, answers):
    from plotly.offline import plot
    import plotly.graph_objs as go
    import numpy as np

    best = min(answers)
    best_idx = answers.index(best)
    traces = [go.Scatter(
        x=times,
        y=answers,
        mode='lines+markers',
        name='solutions over time'
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
        times, answers, best_energy = parse(f.readlines())
        print(','.join(map(str, answers)))
        print(','.join(map(str, times)))
        print('Best: ', best_energy)

        if plot:
            plot_energies(times, answers)
    print('done')


if __name__ == "__main__":
    cli()
