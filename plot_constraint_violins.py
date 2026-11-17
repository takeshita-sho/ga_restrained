"""
Compare the Amplitude and Period distributions the GA reaches under each of its four
constraint modes, as a single panel of four violins per variable. A condition with only a few
rows (--point-max) is drawn as individual points instead.

Inputs are the CSVs written by ga.py -- either the sorted `ga_<mode>_<id>.csv` or, more
usefully for a run that hit its wall clock, the streamed `ga_<mode>_<id>_stream.csv`.
Only rows passing the strict `Is_Oscillatory` screen are plotted: the GA's own viability
gate (Fitness > 0, Period >= 1, Amplitude >= 0.01) admits damped and irregular traces,
and those are not what the comparison is about.

Example:
    python plot_constraint_violins.py \
        --csv "All constraints=results/ga_all_1234_stream.csv" \
        --csv "No LP=results/ga_no-LP_3456_stream.csv" \
        --csv "No LK=results/ga_no-LK_2345_stream.csv" \
        --csv "Neither=results/ga_none_4567_stream.csv" \
        --outdir results
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots  # noqa: F401

plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 36, "axes.titleweight": "bold",
    "axes.labelsize": 32, "axes.labelweight": "bold",
    "xtick.labelsize": 28, "ytick.labelsize": 28,
    "legend.fontsize": 28,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "lines.linewidth": 2,
})

# Violin colours, left to right.
# Colour-blind-safe by lightness: black, yellow, brown, mid grey step from darkest to
# lightest-but-one, so they stay distinct under any colour-vision deficiency.
COLORS = ["black", "#F0E442", "#7B4A12", "#8C8C8C"]

# Violin order, left to right. Labels given via --csv are sorted into this order regardless
# of flag order (unknown labels follow, in the order given), so the colours and positions
# match plot_km_df_violins.py, which keeps the same CANONICAL_ORDER.
CANONICAL_ORDER = ["All constraints", "No LP", "No LK", "Neither"]

# Tick text for each violin, keyed by its --csv label so it cannot drift from the data.
LEGEND_LABELS = {
    "All constraints": r"$K_M^{LK}$ = 15, $K_M^{LP}$ = 39",
    "No LK":           r"$K_M^{LK}$ = free, $K_M^{LP}$ = 39",
    "No LP":           r"$K_M^{LK}$ = 15, $K_M^{LP}$ = free",
    "Neither":         r"$K_M^{LK}$ = free, $K_M^{LP}$ = free",
}


def canonical_sort(labels):
    """Known labels in CANONICAL_ORDER first, then any others in the order given."""
    known = [l for l in CANONICAL_ORDER if l in labels]
    return known + [l for l in labels if l not in CANONICAL_ORDER]

# A stream CSV from a run that was killed mid-write can be ~500 MB, so it is read in
# chunks and filtered down before anything is concatenated. Only these three columns are
# ever needed.
COLUMNS   = ["Period", "Amplitude", "Is_Oscillatory"]
CHUNKSIZE = 1_000_000


def load_oscillatory(path):
    """
    Read one GA result CSV and return the rows passing the strict Is_Oscillatory screen.

    Chunked and column-limited because the streamed files are large, and tolerant of a
    truncated final line -- a run killed by SLURM's wall clock flushes on SIGUSR1 but a
    hard SIGKILL can still leave a partial row.
    """
    # Check the header first: pandas' usecols failure names the missing columns but not
    # the file, and result CSVs written before Is_Oscillatory existed still lie around.
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise SystemExit(
            f"{path}: missing column(s) {missing}. This looks like a result CSV from "
            f"before those columns existed; it has {list(header)}.")

    kept = []
    reader = pd.read_csv(path, usecols=COLUMNS, chunksize=CHUNKSIZE,
                         on_bad_lines="skip", engine="c")
    for chunk in reader:
        # The streamed CSV writes Python bools, so the column arrives as the strings
        # "True"/"False"; the sorted CSV may already be a real bool dtype.
        flag = chunk["Is_Oscillatory"]
        if flag.dtype != bool:
            flag = flag.astype(str).str.strip().str.lower().eq("true")
        keep = chunk.loc[flag, ["Period", "Amplitude"]]
        if not keep.empty:
            kept.append(keep)

    if not kept:
        return pd.DataFrame(columns=["Period", "Amplitude"], dtype=float)
    return pd.concat(kept, ignore_index=True)


def parse_labelled_path(spec):
    """Split a `Label=path` argument. The label becomes the violin's x-tick."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"expected 'Label=path', got {spec!r}")
    label, path = spec.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected 'Label=path', got {spec!r}")
    return label, path


def summarize(frame, var):
    """Per-condition n / median / IQR / range, in panel order, printed with the figures."""
    rows = []
    groups = dict(list(frame.groupby("Condition", observed=False)))
    for label in frame["Condition"].cat.categories:
        values = groups[label][var].to_numpy() if label in groups else np.empty(0)
        if values.size == 0:
            rows.append({"Condition": label, "n": 0, "median": np.nan, "q1": np.nan,
                         "q3": np.nan, "min": np.nan, "max": np.nan})
            continue
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        rows.append({"Condition": label, "n": values.size, "median": med,
                     "q1": q1, "q3": q3, "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def plot_variable(frame, counts, var, args):
    """Draw the four violins for one variable and save the figure."""
    plot_frame, ylabel = frame, var

    if var == "Period":
        if args.max_period is not None:
            plot_frame = plot_frame[plot_frame["Period"] <= args.max_period]
        if args.log_period:
            # Build the violin on log10(Period) rather than setting a log y-scale: a KDE
            # estimated in linear space and then drawn on a log axis misrepresents the
            # density, and GA periods span several decades.
            plot_frame = plot_frame.assign(Period=np.log10(plot_frame["Period"]))
            ylabel = r"$\log_{10}$(Period) [s]"
        else:
            ylabel = "Period [s]"
    else:
        ylabel = r"Amplitude ($A_{mem}$)"

    order = list(counts.index)
    palette = {label: COLORS[i % len(COLORS)] for i, label in enumerate(order)}
    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = plot_frame["Condition"].value_counts()
    sparse = [l for l in order if 0 < sizes.get(l, 0) <= args.point_max]
    violin_frame = plot_frame[~plot_frame["Condition"].isin(sparse)]

    # Conditions with no oscillatory points contribute no rows; seaborn still reserves
    # their slot because Condition is an ordered categorical over all four labels.
    if not violin_frame.empty:
        sns.violinplot(
            data=violin_frame,
            x="Condition",
            y=var,
            order=order,
            cut=0,
            ax=ax,
            width=.8,
            linewidth=1.2,
            hue="Condition",
            hue_order=order,
            palette=palette,
            dodge=False,
            legend=False,
            saturation=1,
            inner="quartile",
            inner_kws={"linewidth": 2.5, "color": "#FFFFFF"},
        )

    # A condition with only a handful of rows has no distribution worth a KDE, so each row
    # is drawn as its own point in the condition's slot instead of a violin. Jitter is
    # seeded so the figure is reproducible.
    rng = np.random.default_rng(0)
    for label in sparse:
        ys = plot_frame.loc[plot_frame["Condition"] == label, var].to_numpy()
        xs = order.index(label) + rng.uniform(-.12, .12, size=ys.size)
        ax.scatter(xs, ys, s=350, color=palette[label], edgecolor="black",
                   linewidth=1.5, zorder=3)
    ax.set_xlim(-.5, len(order) - .5)

    if var == "Amplitude":
        ax.set_ylim(-.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))


    # A mode that found nothing has no violin to draw; say so rather than leave a blank slot.
    for label in order:
        if counts[label] == 0:
            ax.text(order.index(label), .5, "No viable\nsolutions", ha="center",
                    va="center", transform=ax.get_xaxis_transform(), color="#333333")

    ax.set_xticks(range(len(order)))
    ax.set_xlabel("")  # seaborn would otherwise label the axis "Condition"
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls="--", alpha=.3)
    fig.tight_layout()

    suffix = "_log" if (var == "Period" and args.log_period) else ""
    out = os.path.join(args.outdir, f"constraint_{var.lower()}_violin{suffix}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_legend(counts, args):
    """Save the constraint key as its own figure: one swatch per violin in that violin's
    colour, with the K_M settings and n, so the data figures carry no legend."""
    order = list(counts.index)
    handles = [Patch(facecolor=COLORS[i % len(COLORS)], edgecolor="black", linewidth=1.5,
                     label=LEGEND_LABELS.get(label, label) + f"\nn={counts[label]:,}")
               for i, label in enumerate(order)]
    fig = plt.figure(figsize=(12, 2.5))
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False,
               handlelength=1.2, handleheight=1.6, columnspacing=1.5)
    out = os.path.join(args.outdir, "constraint_legend.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Violin comparison of Amplitude/Period across GA constraint modes.")
    parser.add_argument("--csv", action="append", required=True, type=parse_labelled_path,
                        metavar="LABEL=PATH", dest="csvs",
                        help="A labelled GA result CSV; repeat once per constraint mode. "
                             "Violins are drawn in the order the flags are given.")
    parser.add_argument("--variable", choices=("Amplitude", "Period", "both"),
                        default="both", help="Which distribution(s) to plot (default: both).")
    parser.add_argument("--point-max", type=int, default=10,
                        help="A condition with at most this many rows is drawn as individual "
                             "points instead of a violin (default: %(default)s).")
    parser.add_argument("--outdir", default=".", help="Directory for the figures.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the saved figures.")
    parser.add_argument("--max-period", type=float, default=None,
                        help="Drop periods above this value from the Period figure. The "
                             "GA's T=2000 s window bounds what it can resolve, so a long "
                             "tail is mostly poorly-determined.")
    parser.add_argument("--log-period", action="store_true",
                        help="Plot Period as log10(Period); the raw values span decades.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    frames = []
    labels = []
    for label, path in args.csvs:
        data = load_oscillatory(path)
        data["Condition"] = label
        frames.append(data)
        labels.append(label)
        print(f"{label:<20s} {len(data):>10,d} oscillatory rows   ({path})")

    labels = canonical_sort(labels)
    frame = pd.concat(frames, ignore_index=True)
    # Ordered categorical so every mode keeps its slot on the x-axis even when it found
    # nothing -- an empty violin is a result, not a reason to shift the other three.
    frame["Condition"] = pd.Categorical(frame["Condition"], categories=labels, ordered=True)
    counts = frame["Condition"].value_counts().reindex(labels)

    plot_legend(counts, args)

    variables = ["Amplitude", "Period"] if args.variable == "both" else [args.variable]
    for var in variables:
        print(f"\n=== {var} ===")
        print(summarize(frame, var).to_string(index=False))
        plot_variable(frame, counts, var, args)


if __name__ == "__main__":
    main()
