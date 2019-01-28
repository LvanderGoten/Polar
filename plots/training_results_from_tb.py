import os
from argparse import ArgumentParser
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import namedtuple, OrderedDict
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Matplotlib
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Constants
METRICS = ["l1", "l1_phi", "l1_radius"]
METRICS_LATEX = [r"$\mathcal{L}_1$", r"$\mathcal{L}_\phi$", r"$\mathcal{L}_r$"]

# Named tuples
ScalarSummary = namedtuple("ScalarSummary", field_names=["steps", "values"])
HistogramSummary = namedtuple("HistogramSummary", field_names=["steps", "basis_points", "values"])
RunSummary = namedtuple("RunSummary", field_names=["scalar_summaries", "histogram_summaries"])


def read_data_from_event_file(event_fname, horizon):

    # Get accumulator
    accumulator = EventAccumulator(event_fname).Reload()
    tags = accumulator.Tags()

    # Scalars
    scalar_names = tags["scalars"]
    scalar_steps = dict()
    scalar_values = dict()
    for scalar_name in scalar_names:
        scalar_steps[scalar_name] = [scalar_event.step for scalar_event in accumulator.Scalars(scalar_name) if
                                     scalar_event.step < horizon]
        scalar_values[scalar_name] = [scalar_event.value for scalar_event in accumulator.Scalars(scalar_name) if
                                      scalar_event.step < horizon]

    # To ndarrays
    scalar_steps = {k: np.array(v, dtype=np.int32) for k, v in scalar_steps.items()}
    scalar_values = {k: np.array(v, dtype=np.float32) for k, v in scalar_values.items()}

    # Histograms
    histogram_names = tags["distributions"]
    histogram_steps = dict()
    histogram_basis_points = dict()
    histogram_values = dict()
    for histogram_name in histogram_names:
        histogram_steps[histogram_name] = [histogram_event.step for histogram_event in
                                           accumulator.CompressedHistograms(histogram_name) if
                                           histogram_event.step < horizon]
        compressed_histogram_values = [histogram_event.compressed_histogram_values for histogram_event in
                                       accumulator.CompressedHistograms(histogram_name) if
                                       histogram_event.step < horizon]
        histogram_basis_points[histogram_name] = [[v.basis_point for v in compressed_histogram_value] for
                                                  compressed_histogram_value in compressed_histogram_values]
        histogram_values[histogram_name] = [[v.value for v in compressed_histogram_value] for compressed_histogram_value
                                            in compressed_histogram_values]

    # To ndarrays
    histogram_steps = {k: np.array(v, dtype=np.int32) for k, v in histogram_steps.items()}
    histogram_basis_points = {k: np.array(v, dtype=np.float32) for k, v in histogram_basis_points.items()}
    histogram_values = {k: np.array(v, dtype=np.float32) for k, v in histogram_values.items()}

    # To named tuples
    scalar_summaries = {scalar_name: ScalarSummary(steps=scalar_steps[scalar_name],
                                                   values=scalar_values[scalar_name]) for scalar_name in scalar_names}
    histogram_summaries = {histogram_name: HistogramSummary(steps=histogram_steps[histogram_name],
                                                            basis_points=histogram_basis_points[histogram_name],
                                                            values=histogram_values[histogram_name]) for histogram_name
                           in histogram_names}

    return RunSummary(scalar_summaries=scalar_summaries, histogram_summaries=histogram_summaries)


def main():

    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--event_files",
                        help="The binary TensorBoard log files",
                        nargs="+",
                        required=True)

    parser.add_argument("--labels",
                        help="Label of the event runs",
                        nargs="+",
                        required=True)

    parser.add_argument("--horizon",
                        help="How many time steps should be taken",
                        type=int,
                        default=2000)

    # Parse
    args = parser.parse_args()

    # Input assertions
    assert all(os.path.isfile(event_file) for event_file in args.event_files), "TensorBoard log file is faulty!"
    assert len(args.event_files) == len(args.labels), "Event files and labels do not correspond!"

    # Extract run summaries
    run_summaries = OrderedDict(
        (label, read_data_from_event_file(event_fname=event_file, horizon=args.horizon)) for label, event_file in
        zip(args.labels, args.event_files))

    # Epochs vs. loss
    num_metrics = len(METRICS)
    num_runs = len(args.labels)

    fig, ax = plt.subplots(nrows=num_metrics, ncols=1, sharex=True, sharey=True, figsize=(30, 17))
    handles = []
    for metric_id, metric in enumerate(METRICS):
        for run_id, (run_label, run_summary) in enumerate(run_summaries.items()):
            metric_steps, metric_values = run_summary.scalar_summaries[metric]

            # Interpolate
            steps = np.linspace(metric_steps[0], metric_steps[-1])
            metric_interp = splrep(metric_steps, metric_values, k=3)

            handles.append(ax[metric_id].plot(steps, splev(steps, metric_interp), label=run_label, linewidth=6)[0])

    # Legend
    leg = fig.legend(handles[:num_runs],
                     labels=[r"$\lambda = {}$".format(label) for label in args.labels],
                     loc="center right",
                     title=r"$\sigma\ Steepness$")
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    # Titles
    for metric_id, metric_latex in enumerate(METRICS_LATEX):
        ax[metric_id].set_title(metric_latex)
        ax[metric_id].grid(False)

    plt.xlabel("Steps")
    plt.subplots_adjust(right=0.9)
    sns.despine(left=True, bottom=True, right=True)
    plt.savefig("loss.png", transparent=True)


if __name__ == "__main__":
    main()
