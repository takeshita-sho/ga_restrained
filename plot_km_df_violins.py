"""
Compare the Michaelis constants and turnover numbers of the LK and LP enzymatic steps, their
catalytic efficiencies (kcat/Km), and the DF values, that the GA reaches under each of its
four constraint modes, as a single panel of four violins per variable. A condition with only a few
rows (--point-max) is drawn as individual points instead.

The Michaelis constants and catalytic efficiencies are derived from each row's rate
constants,

    Km_LK = (krLK + kcatLK) / kfLK
    Km_LP = (krLP + kcatLP) / kfLP
    kcatKm_LK = kcatLK / Km_LK
    kcatKm_LP = kcatLP / Km_LP

so under a mode that enforces a constraint the corresponding Km is pinned (Km_LK = 15,
Km_LP = 39, up to floating-point noise) and its violin collapses to a line. That is the
point of the comparison: the modes that drop a constraint show where the GA takes Km when
it is left free. The pinned values are drawn as a dashed reference line.

kcatLP is a hardcoded constant in every mode (85.3, never sampled or derived), so kcat_LP
and, whenever the LP constraint also pins Km_LP, kcatKm_LP (= 85.3/39) get the same dashed
reference treatment. kcatLK has no such fixed value in any mode -- it is either derived from
kfLK/krLK or sampled directly, and varies across the population either way -- so kcat_LK and
kcatKm_LK are plotted without a reference line.

Inputs are the CSVs written by ga.py -- either the sorted `ga_<mode>_<id>.csv` or, more
usefully for a run that hit its wall clock, the streamed `ga_<mode>_<id>_stream.csv`.
Only rows passing the strict `Is_Oscillatory` screen are plotted, as in
plot_constraint_violins.py.

Example:
    python plot_km_df_violins.py \
        --csv "All constraints=results/ga_all_1234_stream.csv" \
        --csv "No LP=results/ga_no-LP_3456_stream.csv" \
        --csv "No LK=results/ga_no-LK_2345_stream.csv" \
        --csv "Neither=results/ga_none_4567_stream.csv" \
        --outdir results --log-km --log-df
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots  # noqa: F401

plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "font.size": 36,
    "axes.titlesize": 42, "axes.titleweight": "bold",
    "axes.labelsize": 40, "axes.labelweight": "normal",
    "xtick.labelsize": 36, "ytick.labelsize": 36,
    "legend.fontsize": 34,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "lines.linewidth": 2,
})

# Violin colours, left to right.
# Colour-blind-safe by lightness: black, yellow, brown, mid grey step from darkest to
# lightest-but-one, so they stay distinct under any colour-vision deficiency.
COLORS = ["black", "#F0E442", "#7B4A12", "#8C8C8C"]

# Violin order, left to right. Labels given via --csv are sorted into this order regardless
# of flag order (unknown labels follow, in the order given), so the colours and positions
# match plot_constraint_violins.py, which keeps the same CANONICAL_ORDER and carries the
# tick labels.
CANONICAL_ORDER = ["All constraints", "No LP", "No LK", "Neither"]


def canonical_sort(labels):
    """Known labels in CANONICAL_ORDER first, then any others in the order given."""
    known = [l for l in CANONICAL_ORDER if l in labels]
    return known + [l for l in labels if l not in CANONICAL_ORDER]

# The constrained Km values and the fixed LP turnover number, mirroring KMLK / KMLP /
# KCAT_LP_FIXED in ga.py. Duplicated rather than imported because importing ga.py pulls in
# its solver stack just to read three numbers.
KMLK = 15.0
KMLP = 39.0
# kcatLP is a hardcoded constant in every constraint mode (ga.py never samples or derives
# it), unlike kcatLK, which is either derived from kfLK/krLK or sampled directly and so
# never collapses to one value even when the LK constraint holds. That is why only the LP
# side gets a kcat/kcat-Km reference line below.
KCAT_LP_FIXED = 85.3

# A stream CSV from a run that was killed mid-write can be ~500 MB, so it is read in
# chunks and filtered down before anything is concatenated. Only these columns are needed.
RATE_COLUMNS = ["kfLK", "krLK", "kcatLK", "kfLP", "krLP", "kcatLP", "DF"]
COLUMNS      = RATE_COLUMNS + ["Is_Oscillatory"]
CHUNKSIZE    = 1_000_000

VARIABLES = ["Km_LK", "Km_LP", "DF", "kcat_LK", "kcat_LP", "kcatKm_LK", "kcatKm_LP"]

# Per-variable plot metadata: axis name and unit, title, output stem, constrained reference value,
# and which --log-* flag controls its axis.
META = {
    "Km_LK":     {"name": r"K_{M}^{LK}", "unit": r"[$\mu$M]", "title": r"$K_{M}$ of LK",
                  "stem": "km_lk", "ref": KMLK, "ref_unit": r"$\mu$M",
                  "log_arg": "log_km"},
    "Km_LP":     {"name": r"K_{M}^{LP}", "unit": r"[$\mu$M]", "title": r"$K_{M}$ of LP",
                  "stem": "km_lp", "ref": KMLP, "ref_unit": r"$\mu$M",
                  "log_arg": "log_km"},
    "DF":        {"name": "DF", "unit": "", "title": "DF",
                  "stem": "df", "ref": None, "log_arg": "log_df"},
    "kcat_LK":   {"name": r"k_{cat}^{LK}", "unit": r"[s$^{-1}$]", "title": r"$k_{cat}$ of LK",
                  "stem": "kcat_lk", "ref": None, "log_arg": "log_kcat"},
    "kcat_LP":   {"name": r"k_{cat}^{LP}", "unit": r"[s$^{-1}$]", "title": r"$k_{cat}$ of LP",
                  "stem": "kcat_lp", "ref": KCAT_LP_FIXED, "ref_unit": r"s$^{-1}$",
                  "log_arg": "log_kcat"},
    "kcatKm_LK": {"name": r"k_{cat}/K_{M}^{LK}", "unit": r"[$\mu$M$^{-1}$s$^{-1}$]",
                  "title": r"$k_{cat}/K_{M}$ of LK",
                  "stem": "kcat_km_lk", "ref": None, "log_arg": "log_kcatkm"},
    "kcatKm_LP": {"name": r"k_{cat}/K_{M}^{LP}", "unit": r"[$\mu$M$^{-1}$s$^{-1}$]",
                  "title": r"$k_{cat}/K_{M}$ of LP",
                  "stem": "kcat_km_lp", "ref": KCAT_LP_FIXED / KMLP,
                  "ref_unit": r"$\mu$M$^{-1}$s$^{-1}$", "ref_fmt": ".2f",
                  "log_arg": "log_kcatkm"},
}


def load_oscillatory(path):
    """
    Read one GA result CSV and return Km_LK, Km_LP and DF for the rows passing the strict
    Is_Oscillatory screen.

    Chunked and column-limited because the streamed files are large, and tolerant of a
    truncated final line -- a run killed by SLURM's wall clock flushes on SIGUSR1 but a
    hard SIGKILL can still leave a partial row.
    """
    # Check the header first: pandas' usecols failure names the missing columns but not
    # the file, and result CSVs written before these columns existed still lie around.
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
        rates = chunk.loc[flag, RATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if rates.empty:
            continue
        km_lk = (rates["krLK"] + rates["kcatLK"]) / rates["kfLK"]
        km_lp = (rates["krLP"] + rates["kcatLP"]) / rates["kfLP"]
        keep = pd.DataFrame({
            "Km_LK":     km_lk,
            "Km_LP":     km_lp,
            "DF":        rates["DF"],
            "kcat_LK":   rates["kcatLK"],
            "kcat_LP":   rates["kcatLP"],
            "kcatKm_LK": rates["kcatLK"] / km_lk,
            "kcatKm_LP": rates["kcatLP"] / km_lp,
        })
        # A partially written row can survive on_bad_lines with blank fields.
        keep = keep.replace([np.inf, -np.inf], np.nan).dropna()
        if not keep.empty:
            kept.append(keep)

    if not kept:
        return pd.DataFrame(columns=VARIABLES, dtype=float)
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
    meta = META[var]
    plot_frame, name, ref = frame, meta["name"], meta["ref"]

    use_log = getattr(args, meta["log_arg"])
    if use_log:
        # Build the violin on log10 of the variable rather than setting a log y-scale: a
        # KDE estimated in linear space and then drawn on a log axis misrepresents the
        # density, and both Km (when free) and DF span several decades.
        plot_frame = plot_frame[plot_frame[var] > 0]
        plot_frame = plot_frame.assign(**{var: np.log10(plot_frame[var])})
        name = r"\log_{10}(" + name + ")"
        if ref is not None:
            ref = np.log10(ref)

    order = list(counts.index)
    palette = {label: COLORS[i % len(COLORS)] for i, label in enumerate(order)}
    fig, ax = plt.subplots(figsize=(12, 8))

    # A Km pinned by its constraint has zero spread up to float noise. Its KDE is then a
    # spike tall enough to flatten every other violin, so pinned conditions are pulled out
    # of the violin data and drawn as a flat bar at their value instead.
    sizes = plot_frame["Condition"].value_counts()
    sparse = [l for l in order if 0 < sizes.get(l, 0) <= args.point_max]
    pinned = {}
    for label, values in plot_frame.groupby("Condition", observed=True)[var]:
        if label in sparse:
            continue
        if values.size and np.ptp(values) <= 1e-9 * max(abs(values.median()), 1.0):
            pinned[label] = values.median()
    violin_frame = plot_frame[~plot_frame["Condition"].isin(list(pinned) + sparse)]

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
            # Same maximum width for every violin: the conditions differ in n by orders of
            # magnitude, and area scaling would leave the small ones invisible.
            density_norm="width",
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

    for label, value in pinned.items():
        x = order.index(label)
        ax.hlines(value, x - .4, x + .4, color=palette[label], lw=4)

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

    # A mode that found nothing has no violin to draw; say so rather than leave a blank slot.
    for label in order:
        if counts[label] == 0:
            ax.text(order.index(label), .5, "No viable\nsolutions", ha="center",
                    va="center", transform=ax.get_xaxis_transform(), color="#333333")

    if ref is not None:
        # Black dashed: yellow, brown and grey are all violin colours.
        ax.axhline(ref, ls="--", lw=2, color="black", zorder=0,
                   label=f"Constrained value "
                         f"({meta['ref']:{meta.get('ref_fmt', 'g')}} {meta['ref_unit']})")
        ax.legend(loc="best", frameon=False)

    ax.set_xticks(range(len(order)))
    # Violin names and n appear only on the amplitude figure (plot_constraint_violins.py).
    ax.set_xticklabels([])
    ax.set_xlabel("")  # seaborn would otherwise label the axis "Condition"
    ax.tick_params(axis="x", length=0)
    # Only the variable name is bold; the unit stays regular weight.
    ylabel = r"$\mathbf{" + name + "}$"
    if meta["unit"]:
        ylabel += " " + meta["unit"]
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls="--", alpha=.3)
    fig.tight_layout()

    suffix = "_log" if use_log else ""
    out = os.path.join(args.outdir, f"constraint_{meta['stem']}_violin{suffix}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Violin comparison of Km_LK/Km_LP/DF across GA constraint modes.")
    parser.add_argument("--csv", action="append", required=True, type=parse_labelled_path,
                        metavar="LABEL=PATH", dest="csvs",
                        help="A labelled GA result CSV; repeat once per constraint mode. "
                             "Violins are drawn in the order the flags are given.")
    parser.add_argument("--variable", choices=(*VARIABLES, "Km", "kcat", "kcatKm", "all"),
                        default="all",
                        help="Which distribution(s) to plot: one variable, 'Km' for both "
                             "Michaelis constants, 'kcat' for both turnover numbers, "
                             "'kcatKm' for both catalytic efficiencies, or 'all' (default).")
    parser.add_argument("--point-max", type=int, default=10,
                        help="A condition with at most this many rows is drawn as individual "
                             "points instead of a violin (default: %(default)s).")
    parser.add_argument("--outdir", default=".", help="Directory for the figures.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the saved figures.")
    parser.add_argument("--log-km", action="store_true",
                        help="Plot Km as log10(Km); free Km values span decades.")
    parser.add_argument("--log-df", action="store_true",
                        help="Plot DF as log10(DF); its sampling range is [1e2, 1e4].")
    parser.add_argument("--log-kcat", action="store_true",
                        help="Plot kcat as log10(kcat).")
    parser.add_argument("--log-kcatkm", action="store_true",
                        help="Plot kcat/Km as log10(kcat/Km).")
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

    if args.variable == "all":
        variables = VARIABLES
    elif args.variable == "Km":
        variables = ["Km_LK", "Km_LP"]
    elif args.variable == "kcat":
        variables = ["kcat_LK", "kcat_LP"]
    elif args.variable == "kcatKm":
        variables = ["kcatKm_LK", "kcatKm_LP"]
    else:
        variables = [args.variable]
    for var in variables:
        print(f"\n=== {var} ===")
        print(summarize(frame, var).to_string(index=False))
        plot_variable(frame, counts, var, args)


if __name__ == "__main__":
    main()
