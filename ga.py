import os
import sys

# Force NumPy/SciPy to use only 1 thread per process to prevent CPU thrashing.
# These MUST be set before numpy/scipy are imported for the first time -- the BLAS
# backend reads them at import, so setting them afterwards is silently ignored (and
# would also defeat solve_ode.py's own copy of this guard, since numpy would already
# be loaded by the time that module is imported).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import math
import random
import numpy as np
import pandas as pd
import signal
import time
import multiprocessing

from deap import base, creator, tools, algorithms
from oscillator_models.solve_ode import calc_fitness_math, full_model_ode, process_single_row

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==================== PARAMETER CONSTRAINTS ====================
#
# Two structural constraints can each be enforced or dropped, giving four modes -- see
# `configure_constraints` and the --free-lk / --free-lp flags:
#
#   LK:  kcatLK = KMLK * kfLK - krLK,  with krLK bounded above so kcatLK >= 1e-3
#   LP:  krLP   = KMLP * kfLP - kcatLP,  with kfLP bounded below so krLP >= 1e-3
#
# A dropped constraint turns its derived quantity into an ordinary sampled gene, so the
# gene vector is 10 parameters with both constraints on and 12 with both off.
#
# Free parameters, fully constrained:
#   kfLA, kfLK, krLK, kfLP, kfAP, DF, L, K, P, A
#     --free-lk additionally samples kcatLK,  --free-lp additionally samples krLP
#
# Fixed:
#   kfAK = 0.118,  krAK = 0.0609,  kcatLP = 85.3
#
# Always derived (never sampled in any mode):
#   krLA   = (20/7) * kfLA
#   krAP   = 29 * kfAP

KFAK_FIXED    = 0.118
KRAK_FIXED    = 0.0609
KCAT_LP_FIXED = 85.3

KDLA = 20.0 / 7.0    # krLA   = KDLA * kfLA
KDAP = 29.0          # krAP   = KDAP * kfAP
KMLK = 15.0          # kcatLK = KMLK * kfLK - krLK      (LK constraint)
KMLP = 39.0          # krLP   = KMLP * kfLP - kcatLP    (LP constraint)

# Lower bound on kfLP that makes krLP = KMLP*kfLP - KCAT_LP_FIXED >= 1e-3. Under the LP
# constraint krLP is not a gene at all, so this bound *is* the enforcement -- there is
# nothing left for a rejection test to catch, which is why satisfies_constraints below
# only ever looks at LK.
KFLP_MIN_CONSTRAINED = (KCAT_LP_FIXED + 1e-3) / KMLP   # ~2.187

# Floor on the derived kcatLK. It mirrors the 1e-3 floor the LP constraint places on krLP,
# and matches the bottom of kcatLK's own sampling range when --free-lk makes it a gene, so
# kcatLK spans the same interval in every mode. generateCandidate caps krLK with it, which
# is what makes kcatLK >= KCATLK_MIN true by construction rather than by rejection.
KCATLK_MIN = 1e-3

# Every range the GA might sample, in the canonical gene order. `configure_constraints`
# selects the subset that is free in the requested mode; keeping one table here is what
# stops the constrained and unconstrained variants of a bound from drifting apart.
PARAM_RANGES_ALL = {
    "kfLA":   {"min": 1e-3, "max": 10.0},   # kf  [1e-3, 10]
    "kfLK":   {"min": 1e-3, "max": 10.0},   # kf  [1e-3, 10]
    "krLK":   {"min": 1e-3, "max": 1e3},    # kr  [1e-3, 1e3]
    "kcatLK": {"min": 1e-3, "max": 1e3},    # a gene only under --free-lk
    "kfLP":   {"min": 1e-3, "max": 10.0},   # min raised to KFLP_MIN_CONSTRAINED under LP
    "krLP":   {"min": 1e-3, "max": 1e3},    # a gene only under --free-lp
    "kfAP":   {"min": 1e-3, "max": 10.0},   # kf  [1e-3, 10]
    "DF":     {"min": 1e2,  "max": 1e4},
    "L":      {"min": 1e-2, "max": 1e2},
    "K":      {"min": 1e-3, "max": 1e2},
    "P":      {"min": 1e-3, "max": 1e1},
    "A":      {"min": 1e-2, "max": 1e2},
}

# (FREE_LK, FREE_LP) -> short name used in output filenames and the CSV's Constraints column
MODE_LABELS = {(False, False): "all",   (True, False): "no-LK",
               (False, True):  "no-LP", (True, True):  "none"}

# Set by configure_constraints(). These are module globals rather than Solver state
# because `evaluate_candidate` runs inside forked pool workers and has to reach them
# without being handed a bound method. configure_constraints() must therefore be called
# BEFORE Solver.__init__ creates the pool -- the fork is what carries the configuration
# into the workers.
FREE_LK = FREE_LP = False
MODE_LABEL = MODE_LABELS[(FREE_LK, FREE_LP)]
params_to_optimize = []
parameter_ranges = {}
# Only these two genes can violate kcatLK = KMLK*kfLK - krLK > 0, so only mutations
# touching them need a constraint re-check.
IDX_KFLK = IDX_KRLK = None


def configure_constraints(free_lk=False, free_lp=False):
    """
    Select which structural constraints are enforced, and derive the gene list from that.

    free_lk : sample kcatLK directly instead of deriving it from kfLK/krLK, dropping the
              positivity rejection along with it.
    free_lp : sample krLP directly instead of deriving it from kfLP, and restore kfLP's
              unconstrained lower bound of 1e-3.

    Returns the resulting gene list; the module globals it sets are what the pool workers
    actually read.
    """
    global FREE_LK, FREE_LP, MODE_LABEL, params_to_optimize, parameter_ranges
    global IDX_KFLK, IDX_KRLK

    FREE_LK, FREE_LP = bool(free_lk), bool(free_lp)
    MODE_LABEL = MODE_LABELS[(FREE_LK, FREE_LP)]

    params_to_optimize = (["kfLA", "kfLK", "krLK"]
                          + (["kcatLK"] if FREE_LK else [])
                          + ["kfLP"]
                          + (["krLP"] if FREE_LP else [])
                          + ["kfAP", "DF", "L", "K", "P", "A"])

    # Copy the range dicts -- the kfLP bound below is mutated per-mode, and aliasing it
    # back into PARAM_RANGES_ALL would leak the constrained bound into later calls.
    parameter_ranges = {p: dict(PARAM_RANGES_ALL[p]) for p in params_to_optimize}
    if not FREE_LP:
        parameter_ranges["kfLP"]["min"] = KFLP_MIN_CONSTRAINED

    IDX_KFLK = params_to_optimize.index("kfLK")
    IDX_KRLK = params_to_optimize.index("krLK")

    if not FREE_LK:
        # generateCandidate narrows krLK's upper bound to KMLK*kfLK - KCATLK_MIN, which
        # only works if kfLK has already been drawn and if that bound stays above krLK's
        # floor even at the smallest kfLK. Both hold for the ranges above; asserting them
        # means a later edit to those ranges fails here rather than silently sampling a
        # reversed interval.
        assert IDX_KFLK < IDX_KRLK, "generateCandidate must draw kfLK before krLK"
        krlk_hi_at_kflk_min = KMLK * parameter_ranges["kfLK"]["min"] - KCATLK_MIN
        assert krlk_hi_at_kflk_min > parameter_ranges["krLK"]["min"], (
            f"LK constraint leaves no feasible krLK at kfLK="
            f"{parameter_ranges['kfLK']['min']:g}")

    return params_to_optimize


# Default to the fully constrained configuration, so importing this module -- or forgetting
# the call in a new entry point -- gives a well-defined GA rather than empty globals.
configure_constraints(free_lk=False, free_lp=False)

# ODE settings the GA passes to solve_ode.process_single_row. The integrator, tolerances
# and output grid are solve_ode.py's business, not this file's.
#
# Job 29959062 measured ~2.3 s/solve against the 0.085 s of job 29790204 -- not a
# pathology, but the price of the region the GA now finds: with kcatLK free, the recovered
# solutions have a median period of 66 s, so T=2000 integrates a median of 27 oscillation
# cycles (and up to ~180). That trajectory length, not any single stiff straggler, is
# where the time goes.
#
# SOLVER_TIMEOUT is wall-clock, since that is what process_single_row's timeout event
# measures. Keep it generous: the whole point of this run is the long multi-cycle traces,
# so a tight cap discards exactly the solutions worth having (75.1% of job 29959062's
# points passed the strict oscillator screen, against 0.1% for the pre-kcatLK baseline).
# An earlier version of this file cut it to 1 s on the mistaken belief that stragglers
# dominated; measured cost on real solutions is 3.5-4 s, so that would have thrown away
# the majority of them.
SIM_TIME       = 2000.0
SOLVER_TIMEOUT = 15.0


def _genes(candidate):
    """Name a candidate's genes. The vector's length depends on the active mode."""
    return dict(zip(params_to_optimize, candidate))


def _derived(g):
    """
    kcatLK and krLP, whether they were sampled or follow from an enforced constraint.

    Single source of truth for `evaluate_candidate` and `_build_row` alike, so the values
    handed to the integrator and the values written to the CSV cannot drift.
    """
    kcatLK = g["kcatLK"] if FREE_LK else KMLK * g["kfLK"] - g["krLK"]
    krLP   = g["krLP"]   if FREE_LP else KMLP * g["kfLP"] - KCAT_LP_FIXED
    return kcatLK, krLP


def _build_ode_params(g, kcatLK, krLP):
    """Assemble the 13-element ODE parameter vector from the named free params."""
    return np.array([g["kfLA"], KDLA * g["kfLA"], g["kfLK"], g["krLK"], kcatLK,
                     g["kfLP"], krLP, KCAT_LP_FIXED,
                     KFAK_FIXED, KRAK_FIXED,
                     g["kfAP"], KDAP * g["kfAP"], g["DF"]], dtype=np.float64)


def satisfies_constraints(candidate):
    """
    Whether a candidate is inside the feasible set while the LK constraint is on.

    The bound is KCATLK_MIN rather than a bare > 0 so that this agrees exactly with what
    generateCandidate constructs -- otherwise mutation and crossover could walk the
    population into a sliver of kcatLK that the initial sampling never produces.

    There is deliberately no LP counterpart: with the LP constraint enforced krLP is not a
    gene, and kfLP's raised lower bound already guarantees krLP >= 1e-3, so no candidate
    can violate it and no test would ever fire.
    """
    if FREE_LK:
        return True
    return KMLK * candidate[IDX_KFLK] - candidate[IDX_KRLK] >= KCATLK_MIN



# ==================== FITNESS FUNCTION ====================

def evaluate_candidate(candidate):
    """
    Top-level fitness function (module-level for multiprocessing pickling).

    Returns (cost, period, amplitude, is_oscillatory, elapsed_seconds, timed_out), NOT a
    DEAP fitness tuple: FitnessMax carries a single weight, so `Solver._evaluate` is
    responsible for assigning `ind.fitness.values = (cost,)` and stashing the first four
    on the individual as `ind.metrics`. Consequently this toolbox is NOT drop-in
    compatible with stock `algorithms.eaSimple` / `eaMuPlusLambda`, which would try to
    assign all of them and trip DEAP's length assertion.

    The last two values are instrumentation, timed around `process_single_row` from the
    outside so `solve_ode.py` needs no changes. Job 29959062 spent 70 h and 3,239
    CPU-hours without ever revealing its per-solve cost, which is the whole reason it was
    hard to diagnose.

    `timed_out` is inferred rather than reported: `process_single_row` folds a timeout and
    a genuine zero into the same `Fitness = 0`, so the only signal available here is that
    the call consumed very nearly its whole budget. Wall-clock is the right clock for that
    comparison because that is what the timeout event inside `process_single_row`
    measures. The threshold is deliberately loose -- this is a diagnostic counter, not a
    control input.

    Candidates violating an enforced constraint are penalised with fitness 0.
    """
    g = _genes(candidate)
    kcatLK, krLP = _derived(g)

    # Safety net. Sampling builds the constraints in and mutation/crossover revert any
    # violation, so
    # this should never fire -- but a negative rate reaching the integrator would produce
    # a plausible-looking trace rather than an error, which is worth one comparison.
    if kcatLK <= 0 or krLP <= 0:
        return (0.0, 0.0, 0.0, False, 0.0, False)

    params = _build_ode_params(g, kcatLK, krLP)
    params = np.concatenate(
        [params, np.array([g["L"], g["K"], g["P"], g["A"]], dtype=np.float64)])

    t0 = time.perf_counter()
    metrics = process_single_row(params, T=SIM_TIME, timeout=SOLVER_TIMEOUT, method='LSODA')
    elapsed = time.perf_counter() - t0

    return (float(metrics[0]), float(metrics[1]), float(metrics[2]), bool(metrics[3]),
            elapsed, elapsed >= 0.95 * SOLVER_TIMEOUT)


# ==================== SOLVER CLASS ====================

class Solver:
    def __init__(self, populationSize=10000, NGEN=100, outfileName="ga_oscillator_results.csv",
                 restarts=4, genBudget=100, patience=10, seed=None,
                 hofSize=50):
        self.populationSize = populationSize
        self.NGEN           = NGEN        # max generations within a single restart
        self.genBudget      = genBudget   # total generations across all restarts
        self.restarts       = restarts    # max independent restarts
        self.patience       = patience    # barren generations before restarting
        self.outfileName    = outfileName
        self.threshold      = 0.0   # keep solutions with fitness > 0
        self.indpb          = 0.2   # per-gene MUTATION probability
        self.eta            = 2.0   # polynomial mutation distribution index
        self.mutScale       = 0.25  # shrinks the polynomial step for finer local search
        self.seed           = seed

        # Two RNGs are in play -- `random` drives DEAP's operators (varAnd, tournament
        # selection), `np.random` drives the custom sampling and mutation -- so both
        # need seeding for a run to be reproducible.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Guard against re-creating DEAP types if Solver is instantiated multiple times
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Candidate"):
            creator.create("Candidate", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("candidate",  self.generateCandidate)
        self.toolbox.register("population", self.generate_random_population)
        self.toolbox.register("mutate",     self.mutateCandidate,
                              indpb=self.indpb, eta=self.eta, scale=self.mutScale)
        self.toolbox.register("select",     tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate",   evaluate_candidate)
        self.toolbox.register("mate",       self.mate_and_validate)

        # Best-by-fitness individuals regardless of the viability gate. Without this the
        # single best genotype the GA finds is unrecoverable whenever it fails the
        # period/amplitude screen, which is exactly what happened on the baseline run
        # (best-in-generation 0.2528 vs best written to CSV 0.1292).
        self.hof = tools.HallOfFame(hofSize)

        # Streaming output. Job 29959062 ran 68 generations and accumulated 1,630,180
        # viable points, then hit its wall-clock limit and wrote nothing at all -- the
        # CSVs are only produced after every restart finishes, and SIGTERM does not run
        # a `finally`. Those points were only recovered by dumping the live process's
        # memory. Every viable point is now appended as it is found, so a killed run
        # leaves usable output on disk.
        self.streamFileName = self.outfileName.replace(".csv", "_stream.csv")
        self._stream = None
        self._writer = None

        # sched_getaffinity, not cpu_count: cpu_count reports every CPU on the node and
        # ignores the cgroup SLURM puts the job in, so a run given --cpus-per-task=8 on a
        # shared node would fork 48 workers onto 8 cores. Four constraint modes are now
        # run as four jobs rather than one, so co-scheduling is likelier than it was.
        self.pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))
        # Small chunks: solve times vary by orders of magnitude, so the default
        # chunksize (~5200 at a 1M population) load-balances badly.
        self.toolbox.register("map", self.pool.map, chunksize=100)

    def generate_random_population(self, n):
        """Generates the initial population by independent log-uniform sampling, with
        the active structural constraints built into the draw."""
        print("Generating initial population via random sampling...")
        population = [self.generateCandidate() for _ in range(n)]
        print(f"Successfully generated {n} random individuals.")
        return population

    def generateCandidate(self):
        """
        Draw one candidate log-uniformly per gene, constructed to satisfy the active
        constraints rather than tested against them afterwards.

        Under the LK constraint krLK's upper bound is narrowed to KMLK*kfLK - KCATLK_MIN
        once kfLK has been drawn, so kcatLK >= KCATLK_MIN holds by construction and no
        draw is ever wasted. The LP constraint needs nothing here: krLP is not a gene and
        kfLP's raised lower bound already guarantees krLP >= 1e-3.

        This replaces a rejection loop that discarded 48% of its draws. Note that the two
        are not the same distribution, and the difference is in this one's favour: keeping
        only the feasible members of a joint log-uniform draw tilted the kfLK marginal
        towards large values, because a large kfLK leaves far more feasible krLK than a
        small one (86% of krLK's range at kfLK=10, against 20% at kfLK=1e-3). Drawing krLK
        from its conditional range instead leaves kfLK log-uniform over its full range.
        """
        candidate = []
        for param_name in params_to_optimize:
            low  = parameter_ranges[param_name]["min"]
            high = parameter_ranges[param_name]["max"]
            if param_name == "krLK" and not FREE_LK:
                # kfLK is already in `candidate` -- configure_constraints asserts the
                # ordering, and that this bound stays above `low` for every legal kfLK.
                high = min(high, KMLK * candidate[IDX_KFLK] - KCATLK_MIN)
            candidate.append(10.0 ** np.random.uniform(math.log10(low), math.log10(high)))

        return creator.Candidate(candidate)

    def mutateCandidate(self, candidate, indpb, eta, scale):
        """
        Polynomial-bounded mutation in log10 space, with per-gene constraint rejection.

        The polynomial step is self-limiting: delta_q is derived from the gene's own
        distance to each bound, so a mutated value approaches a bound asymptotically
        but never overshoots it. That is what removes the boundary pile-up the previous
        `max(lo, min(new, hi))` clamp produced -- on the baseline run 32% of accepted
        solutions sat on DF's ceiling and 20% on kfAP's floor purely as a clamp
        artifact. `scale` shrinks the step further so late generations can refine
        rather than jump up to a full decade per gene.

        If a mutation breaks the LK constraint (when it is enforced), only that gene is
        reverted; the rest of the individual is preserved.
        """
        for idx in range(len(candidate)):
            if np.random.rand() >= indpb:
                continue

            old_val = candidate[idx]
            param_name = params_to_optimize[idx]
            xl = math.log10(parameter_ranges[param_name]["min"])
            xu = math.log10(parameter_ranges[param_name]["max"])
            x  = math.log10(old_val)
            span = xu - xl

            delta1 = (x - xl) / span
            delta2 = (xu - x) / span
            rand = np.random.rand()
            mut_pow = 1.0 / (eta + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
                delta_q = 1.0 - val ** mut_pow

            x += delta_q * span * scale
            # Safety net only -- the polynomial step already respects the bounds, so
            # this should never actually bind.
            candidate[idx] = 10.0 ** min(max(x, xl), xu)

            # Only kfLK/krLK can break the constraint, so re-checking the other genes is
            # wasted work -- and under --free-lk satisfies_constraints is a no-op anyway.
            if idx in (IDX_KFLK, IDX_KRLK) and not satisfies_constraints(candidate):
                candidate[idx] = old_val

        return (candidate,)

    def mate_and_validate(self, ind1, ind2):
        # 1. Save backups of the parents before DEAP modifies them in-place
        parent1_backup = ind1[:]
        parent2_backup = ind2[:]

        # 2. Perform the crossover
        tools.cxTwoPoint(ind1, ind2)

        # 3. Check constraints. If a child is invalid, revert it to its parent's state.
        if not satisfies_constraints(ind1):
            ind1[:] = parent1_backup

        if not satisfies_constraints(ind2):
            ind2[:] = parent2_backup

        return ind1, ind2

    # ---- streaming output & crash safety ----

    @staticmethod
    def _fieldnames():
        """Column order of the streamed CSV, derived from `_build_row` so the two cannot drift."""
        return list(Solver._build_row([0.0] * len(params_to_optimize), 0.0, 0.0, 0.0, False))

    def _open_stream(self):
        self._stream = open(self.streamFileName, "w", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._stream.flush()
        print(f"Streaming viable points to {self.streamFileName} as they are found.", flush=True)

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()
            except Exception:
                pass
            # Drop the writer too: `_record` gates on it, and a live writer over a closed
            # file would raise rather than skip.
            self._stream = None
            self._writer = None

    def install_signal_handlers(self):
        """
        Flush results and exit on SIGTERM/SIGUSR1 rather than dying silently.

        Pair with `#SBATCH --signal=B:USR1@600` so SLURM gives 10 minutes' warning before
        the wall clock, which is what job 29959062 needed and did not have. `os._exit`
        skips atexit and the pool teardown deliberately -- joining 48 mid-solve workers
        during a shutdown is exactly the wrong time to block.
        """
        def _bail(signum, frame):
            print(f"\n[signal {signum}] flushing results and exiting.", flush=True)
            self._close_stream()
            try:
                self.write_hallOfFame()
            except Exception as exc:
                print(f"  hall-of-fame write failed: {exc!r}", flush=True)
            sys.stdout.flush()
            os._exit(0)

        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(sig, _bail)

    # ---- evaluation ----

    def _evaluate(self, individuals):
        """
        Solve the ODE for `individuals`, cache the metrics, set scalar fitness.

        Returns the per-solve (cpu_seconds, timed_out) pairs so the caller can report the
        cost distribution. Without this the run is a black box: job 29959062 gave no way
        to tell 2.3 s/solve from 0.085 s/solve until its memory was dumped.
        """
        if not individuals:
            return []
        timings = []
        for ind, m in zip(individuals, self.toolbox.map(self.toolbox.evaluate, individuals)):
            ind.metrics = (float(m[0]), float(m[1]), float(m[2]), bool(m[3]))
            ind.fitness.values = (ind.metrics[0],)
            timings.append((m[4], m[5]))
        return timings

    @staticmethod
    def _timing_summary(timings):
        """
        One-line cost distribution for a generation's solves.

        Times are wall-clock per solve. Because each pool worker is a dedicated process
        pinned to its own core for the whole solve, summing them is a good estimate of
        core-hours spent -- compare it against the elapsed generation time times the core
        count to see how well the pool is load-balancing.
        """
        if not timings:
            return "solves=0"
        t = np.fromiter((x[0] for x in timings), dtype=np.float64, count=len(timings))
        total = float(t.sum())
        n_to = sum(1 for x in timings if x[1])
        to_time = float(sum(x[0] for x in timings if x[1]))
        share = (100.0 * to_time / total) if total > 0 else 0.0
        return (f"s/solve med={np.median(t):.2f} p90={np.percentile(t, 90):.2f} "
                f"p99={np.percentile(t, 99):.2f} max={t.max():.2f} "
                f"gen_cost={total / 3600.0:.2f}core-h "
                f"timeouts={n_to} ({100.0 * n_to / len(t):.2f}%, {share:.0f}% of time)")

    def _record(self, individuals, seen, points, fitnesses):
        """
        File every individual clearing the viability gate, returning the count of new
        ones. Membership is tested against a set of parameter tuples rather than the
        old `if ind not in all_viable_points` linear scan, which was O(n^2).

        New points are also appended to the stream file immediately, so the run's output
        survives a wall-clock kill.
        """
        new = 0
        for ind in individuals:
            cost, period, amplitude, is_osc = ind.metrics
            if cost > self.threshold and period >= 1.0 and amplitude >= 0.01:
                sig = tuple(ind)
                if sig not in seen:
                    seen.add(sig)
                    points.append(list(ind))
                    fitnesses.append([cost, period, amplitude, is_osc])
                    new += 1
                    if self._writer is not None:
                        self._writer.writerow(
                            self._build_row(ind, cost, period, amplitude, is_osc))
        if new and self._stream is not None:
            self._stream.flush()
        return new

    def _run_one(self, restart_idx, gen_budget, seen, points, fitnesses):
        """
        Run one independent GA from a fresh population. Returns generations consumed.

        Stops early once `patience` consecutive generations contribute no new viable
        point. On the baseline run the population had collapsed by generation 42 and
        the remaining 57 generations -- 57M ODE solves, ~29 h of a 50 h job -- produced
        nothing at all, while the scalar fitness improved by +0.0%. Rolling that budget
        into a fresh restart samples a different basin instead.
        """
        population = self.toolbox.population(n=self.populationSize)

        # Evaluate the initial population. Skipping this used to mean (a) the most
        # diverse sample of the entire run was never recorded, and (b) ~12.5% of the
        # first selected generation carried an invalid fitness -- which in DEAP only
        # ever wins a tournament when all three aspirants are also invalid.
        timings = self._evaluate(population)
        self.hof.update(population)
        new = self._record(population, seen, points, fitnesses)
        best = max(ind.metrics[0] for ind in population)
        print(f"  [restart {restart_idx}] gen   0 (initial) : best={best:.6f}  "
              f"new={new:5d}  total={len(points)}  {self._timing_summary(timings)}",
              flush=True)

        gens_used = 1
        stagnant = 0
        # The initial population above already counts as one evaluated generation, so
        # this loop contributes at most NGEN-1 more.
        for gen in range(1, min(self.NGEN, gen_budget)):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.6, mutpb=0.4)

            # varAnd leaves (1-cxpb)*(1-mutpb) = 24% of offspring untouched. Those keep
            # a valid fitness and their cached metrics, so re-solving them is pure
            # waste -- roughly 12 h of the 50 h baseline run.
            pending = [ind for ind in offspring if not ind.fitness.valid]
            timings = self._evaluate(pending)

            self.hof.update(offspring)
            new = self._record(offspring, seen, points, fitnesses)
            population = self.toolbox.select(population + offspring, k=self.populationSize)

            best = max(ind.metrics[0] for ind in offspring)
            print(f"  [restart {restart_idx}] gen {gen:3d}           : best={best:.6f}  "
                  f"new={new:5d}  total={len(points)}  "
                  f"solved={len(pending)}/{len(offspring)}  "
                  f"{self._timing_summary(timings)}", flush=True)

            gens_used += 1
            stagnant = 0 if new else stagnant + 1
            if stagnant >= self.patience:
                print(f"  [restart {restart_idx}] no new viable points for {self.patience} "
                      f"generations - abandoning this basin.")
                break

        return gens_used

    def findNominalValues(self):
        """Run restarts until the generation budget is spent. Returns viable points."""
        seen, points, fitnesses = set(), [], []

        print("BEGIN main GA loop.")
        remaining = self.genBudget
        for restart_idx in range(self.restarts):
            if remaining <= 0:
                break
            print(f"=== Restart {restart_idx}  ({remaining} generations of budget left) ===")
            remaining -= self._run_one(restart_idx, remaining, seen, points, fitnesses)

        return points, fitnesses

    @staticmethod
    def _build_row(opt, fitness_val, period_val, amplitude_val, is_osc):
        """
        Expand a free-parameter vector into the full ODE parameter set for output.

        The column set is identical in all four constraint modes -- a quantity that is not
        a gene in the active mode is filled in from its constraint -- so CSVs from
        different modes can be compared, or concatenated, with no reshaping.
        """
        g = _genes(opt)
        kcatLK, krLP = _derived(g)
        return {
            'Fitness':        fitness_val,
            'Period':         period_val,
            'Amplitude':      amplitude_val,
            # Strict oscillator verdict from calculate_oscillator_metrics: this carries
            # the damped (screen 3) and regularity (screen 4) tests, which the GA's own
            # period/amplitude gate does not apply. Recorded rather than enforced, so
            # filtering on it stays a post-processing choice.
            'Is_Oscillatory': bool(is_osc),
            # Which constraints the run that produced this row enforced.
            'Constraints':    MODE_LABEL,
            # Model parameters. Whether a given one was sampled or derived depends on the
            # mode; the Constraints column above is what records that.
            'kfLA':   g['kfLA'],
            'kfLK':   g['kfLK'],
            'krLK':   g['krLK'],
            'kfLP':   g['kfLP'],
            'krLP':   krLP,
            'kfAP':   g['kfAP'],
            'DF':     g['DF'],
            'L':      g['L'],
            'K':      g['K'],
            'P':      g['P'],
            'A':      g['A'],
            # Derived / fixed (full ODE parameter set)
            'krLA':   KDLA * g['kfLA'],
            'kcatLK': kcatLK,
            'kcatLP': KCAT_LP_FIXED,
            'kfAK':   KFAK_FIXED,
            'krAK':   KRAK_FIXED,
            'krAP':   KDAP * g['kfAP'],
        }

    def write_sortedParms(self, viablePoints, viableFitness):
        """Sort viable solutions by descending fitness and write full parameter set to CSV."""
        idx_sorted = np.argsort([-vf[0] for vf in viableFitness])

        rows = []
        for rank, idx in enumerate(idx_sorted):
            fitness_val, period_val, amplitude_val, is_osc = viableFitness[idx]
            row = self._build_row(viablePoints[idx], fitness_val, period_val, amplitude_val, is_osc)
            rows.append({'Rank': rank, **row})

        # Emit the header even on an empty result set, so downstream readers get an
        # empty table rather than a zero-byte file they cannot parse.
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['Rank', *self._build_row([0.0] * len(params_to_optimize), 0.0, 0.0, 0.0, False)])
        df.to_csv(self.outfileName, index=False)
        n_osc = sum(1 for r in rows if r['Is_Oscillatory'])
        print(f"Wrote {len(rows)} viable solutions to {self.outfileName} "
              f"({n_osc} of them pass the strict Is_Oscillatory screen)")

    def write_hallOfFame(self):
        """Write the best-by-fitness individuals, whether or not they were viable."""
        if not len(self.hof):
            return
        outfile = self.outfileName.replace(".csv", "_hof.csv")
        rows = [{'Rank': rank, **self._build_row(list(ind), *ind.metrics)}
                for rank, ind in enumerate(self.hof)]
        pd.DataFrame(rows).to_csv(outfile, index=False)
        print(f"Wrote {len(rows)} best-by-fitness solutions to {outfile}")

    def run(self):
        """Run the full GA pipeline and write results."""
        self.install_signal_handlers()
        self._open_stream()
        try:
            viablePoints, viableFitness = self.findNominalValues()
        finally:
            # Close in a finally so a crash mid-run does not leak the worker pool, and
            # so the streamed results are on disk even if the sorted write never happens.
            self._close_stream()
            self.pool.close()
            self.pool.join()

        self.write_sortedParms(viablePoints, viableFitness)
        self.write_hallOfFame()
        print("GA complete.", flush=True)
        return viablePoints, viableFitness


# ==================== MAIN ====================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for the oscillator model. The LK and LP structural "
                    "constraints can each be switched off, giving four modes; both are "
                    "enforced unless a flag says otherwise.")
    parser.add_argument("--free-lk", action="store_true",
                        help="Drop the LK constraint: sample kcatLK as a free gene instead "
                             f"of deriving it as {KMLK:g}*kfLK - krLK > 0.")
    parser.add_argument("--free-lp", action="store_true",
                        help="Drop the LP constraint: sample krLP as a free gene instead of "
                             f"deriving it as {KMLP:g}*kfLP - {KCAT_LP_FIXED:g}, and restore "
                             "kfLP's unconstrained lower bound of 1e-3.")
    # Sizing. Job 29959062 reached only generation 68 of 100 in its full 3-day allocation
    # at ~2.3 s/solve. Read the actual cost off the per-generation `s/solve` line in the
    # log and re-size from it rather than trusting any default here:
    #     generations * population * 0.76 * mean_solve_s / n_cores <= wall_seconds
    # (the 0.76 is varAnd's clone-skip rate -- 24% of offspring arrive unmodified with a
    # still-valid fitness and are not re-solved).
    parser.add_argument("--pop", type=int, default=100_000,
                        help="Population size (default: %(default)s).")
    parser.add_argument("--gens", type=int, default=100,
                        help="Max generations within a single restart (default: %(default)s).")
    parser.add_argument("--gen-budget", type=int, default=100,
                        help="Total generations across all restarts -- the compute cap "
                             "(default: %(default)s).")
    parser.add_argument("--restarts", type=int, default=4,
                        help="Max independent restarts within that budget (default: %(default)s).")
    parser.add_argument("--patience", type=int, default=10,
                        help="Abandon a basin after this many barren generations "
                             "(default: %(default)s).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed. Pass the same value to every constraint mode when "
                             "the runs are meant to be compared.")
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory for the result CSVs (default: this script's directory).")
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = _parse_args()

    # Before Solver.__init__, which creates the worker pool: the fork is what carries this
    # configuration into the workers that run evaluate_candidate.
    configure_constraints(free_lk=args.free_lk, free_lp=args.free_lp)

    # Pre-compile Numba functions in the main process before forking workers.
    # On Linux (fork), child processes inherit the compiled code, avoiding
    # simultaneous recompilation deadlock across all cores.
    print("Pre-compiling Numba functions to prevent worker deadlock...")
    _dummy_Y     = np.random.rand(100)
    _dummy_peaks = np.array([10, 20, 30])
    _ = calc_fitness_math(_dummy_peaks, _dummy_Y)
    _dummy_y      = np.ones(16)
    _dummy_params = np.ones(13)
    _ = full_model_ode(0.0, _dummy_y, _dummy_params)
    print("Pre-compilation complete.")

    os.makedirs(args.outdir, exist_ok=True)
    random_id = np.random.randint(1, 10000)
    outfile   = os.path.join(args.outdir, f"ga_{MODE_LABEL}_{random_id}.csv")
    print(f"Constraints     : {MODE_LABEL}  "
          f"(LK {'FREE' if FREE_LK else 'enforced'}, LP {'FREE' if FREE_LP else 'enforced'})")
    print(f"Population size : {args.pop}")
    print(f"Generations     : {args.gens} max per restart, {args.gen_budget} total budget")
    print(f"Restarts        : up to {args.restarts}, patience {args.patience} barren generations")
    print(f"Seed            : {args.seed}")
    print(f"Output file     : {outfile}")
    print(f"CPU cores       : {len(os.sched_getaffinity(0))} allocated "
          f"(node has {multiprocessing.cpu_count()})")
    print(f"Free parameters : {params_to_optimize}")
    print(f"kfLP range      : [{parameter_ranges['kfLP']['min']:g}, "
          f"{parameter_ranges['kfLP']['max']:g}]")
    print(f"ODE settings    : T={SIM_TIME:g}, timeout={SOLVER_TIMEOUT:g} s "
          f"(integrator and tolerances are solve_ode.process_single_row's)")
    print(flush=True)

    solver = Solver(populationSize=args.pop, NGEN=args.gens, outfileName=outfile,
                    restarts=args.restarts,
                    genBudget=args.gen_budget, patience=args.patience, seed=args.seed)
    viablePoints, viableFitness = solver.run()

    if viableFitness:
        best_idx = max(range(len(viableFitness)), key=lambda i: viableFitness[i][0])
        best_fit = viableFitness[best_idx]
        print(f"\nBest viable fitness : {best_fit[0]:.6f} "
              f"(period {best_fit[1]:.1f} s, amplitude {best_fit[2]:.4f}, "
              f"Is_Oscillatory={bool(best_fit[3])})")
        print(f"Best viable params  : {dict(zip(params_to_optimize, viablePoints[best_idx]))}")
    else:
        print("\nNo viable solutions found (all fitness <= 0).")

    if len(solver.hof):
        top = solver.hof[0]
        print(f"\nBest fitness overall (viable or not) : {top.metrics[0]:.6f} "
              f"(period {top.metrics[1]:.1f} s, amplitude {top.metrics[2]:.4f}, "
              f"Is_Oscillatory={top.metrics[3]})")
        print(f"Best overall params : {dict(zip(params_to_optimize, list(top)))}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.2f} minutes")
