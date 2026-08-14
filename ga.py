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

import math
import random
import numpy as np
import pandas as pd
import time
import multiprocessing

from scipy.stats import qmc
from deap import base, creator, tools, algorithms
from oscillator_models.solve_ode import calc_fitness_math, full_model_ode, process_single_row

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==================== PARAMETER CONSTRAINTS ====================
#
# Free parameters (sampled by GA):
#   kfLA, kfLK, krLK, kfLP, krLP, kfAP, DF, L, K, P, A
#
# Fixed:
#   kfAK = 0.118,  krAK = 0.0609,  kcatLP = 85.3
#
# Derived (computed from free params, never directly sampled):
#   krLA   = (20/7) * kfLA
#   kcatLK = 15 * kfLK - krLK          (penalised to 0 if <= 0)
#   krAP   = 29 * kfAP

KFAK_FIXED    = 0.118
KRAK_FIXED    = 0.0609
KCAT_LP_FIXED = 85.3

KDLA = 20.0 / 7.0   # krLA   = KDLA * kfLA
KDAP = 29.0          # krAP   = KDAP * kfAP
KMLK  = 15.0

parameter_ranges = {
    "kfLA": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "kfLK": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "krLK": {"min": 1e-3,      "max": 1e3},    # kr  [1e-3, 1e3]
    "kfLP": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "krLP": {"min": 1e-3,      "max": 1e3},    # kr  [1e-3, 1e3]
    "kfAP": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "DF":   {"min": 1e2,      "max": 1e4},
    "L":    {"min": 1e-2,      "max": 1e2},
    "K":    {"min": 1e-3,      "max": 1e2},
    "P":    {"min": 1e-3,      "max": 1e1},
    "A":    {"min": 1e-2,      "max": 1e2},
}

params_to_optimize = ["kfLA", "kfLK", "krLK", "kfLP", "krLP", "kfAP", "DF", "L", "K", "P", "A"]

# Only these two genes can violate kcatLK = 15*kfLK - krLK > 0, so only mutations
# touching them need a constraint re-check.
IDX_KFLK, IDX_KRLK = params_to_optimize.index("kfLK"), params_to_optimize.index("krLK")

# ODE settings used by the GA. Deliberately shorter than solve_ode.py's defaults: the
# fitness only needs enough trace to resolve a handful of periods, and the timeout is
# sized against the ~0.09 s mean solve rather than left at the 600 s bulk-run default.
SIM_TIME       = 2000.0
SOLVER_TIMEOUT = 15.0


def _build_ode_params(kfLA, kfLK, krLK, kfLP, krLP, kfAP, DF):
    """Assemble the 13-element ODE parameter vector from the free rate/DF params."""
    krLA   = KDLA * kfLA
    kcatLK = KMLK * kfLK - krLK
    krAP   = KDAP * kfAP
    return np.array([kfLA, krLA, kfLK, krLK, kcatLK,
                     kfLP, krLP, KCAT_LP_FIXED,
                     KFAK_FIXED, KRAK_FIXED,
                     kfAP, krAP, DF], dtype=np.float64)


def satisfies_constraints(candidate):
    """kcatLK = 15*kfLK - krLK must be strictly positive."""
    return KMLK * candidate[IDX_KFLK] - candidate[IDX_KRLK] > 0


# ==================== FITNESS FUNCTION ====================

def evaluate_candidate(candidate):
    """
    Top-level fitness function (module-level for multiprocessing pickling).

    Returns the raw metrics 4-tuple (cost, period, amplitude, is_oscillatory), NOT a
    DEAP fitness tuple: FitnessMax carries a single weight, so `Solver._evaluate` is
    responsible for assigning `ind.fitness.values = (cost,)` and stashing the other
    three on the individual as `ind.metrics`. Consequently this toolbox is NOT drop-in
    compatible with stock `algorithms.eaSimple` / `eaMuPlusLambda`, which would try to
    assign all four values and trip DEAP's length assertion.

    Candidates with kcatLK <= 0 are penalised with fitness 0.
    """
    kfLA, kfLK, krLK, kfLP, krLP, kfAP, DF, L, K, P, A = candidate

    kcatLK = KMLK * kfLK - krLK
    if kcatLK <= 0:
        return (0.0, 0.0, 0.0, False)

    params = _build_ode_params(kfLA, kfLK, krLK, kfLP, krLP, kfAP, DF)
    params = np.concatenate([params, np.array([L, K, P, A], dtype=np.float64)])
    metrics = process_single_row(params, T=SIM_TIME, timeout=SOLVER_TIMEOUT)

    return (float(metrics[0]), float(metrics[1]), float(metrics[2]), bool(metrics[3]))


# ==================== SOLVER CLASS ====================

class Solver:
    def __init__(self, populationSize=10000, NGEN=100, outfileName="ga_oscillator_results.csv",
                 sampling="lhs", restarts=4, genBudget=100, patience=10, seed=None,
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
        self.sampling       = sampling.lower()  # "lhs" or "random"
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
        if self.sampling == "lhs":
            self.toolbox.register("population", self.generate_lhs_population)
        elif self.sampling == "random":
            self.toolbox.register("population", self.generate_random_population)
        else:
            raise ValueError(f"Unknown sampling method '{self.sampling}'; expected 'lhs' or 'random'.")
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

        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        # Small chunks: solve times vary by orders of magnitude, so the default
        # chunksize (~5200 at a 1M population) load-balances badly.
        self.toolbox.register("map", self.pool.map, chunksize=100)

    def generate_lhs_population(self, n):
        """
        Generate the initial population via Latin Hypercube Sampling in log10 space.

        Caveat: the kcatLK constraint rejects ~47% of any hypercube drawn over the full
        box, and the shortfall is topped up from further batches, so the result is only
        approximately stratified -- it is not a true Latin hypercube. It is still a
        better-spread start than independent sampling.
        """
        print("Generating initial population via Latin Hypercube Sampling...")
        dim = len(params_to_optimize)
        sampler = qmc.LatinHypercube(d=dim, seed=self.seed)

        log_lower = [math.log10(parameter_ranges[p]["min"]) for p in params_to_optimize]
        log_upper = [math.log10(parameter_ranges[p]["max"]) for p in params_to_optimize]

        valid = []
        while len(valid) < n:
            linear = 10.0 ** qmc.scale(sampler.random(n=n), log_lower, log_upper)
            # Vectorised constraint mask -- the old row-by-row Python loop over 2n
            # numpy rows was pure overhead.
            keep = linear[(KMLK * linear[:, IDX_KFLK] - linear[:, IDX_KRLK]) > 0]
            valid.extend(creator.Candidate(row) for row in keep[: n - len(valid)].tolist())

        print(f"Successfully generated {n} LHS individuals.")
        return valid

    def generate_random_population(self, n):
        """Generates the initial population by independent random log-uniform sampling,
        respecting structural constraints."""
        print("Generating initial population via random sampling...")
        population = [self.generateCandidate() for _ in range(n)]
        print(f"Successfully generated {n} random individuals.")
        return population

    def generateCandidate(self):
        """Sample each free parameter uniformly in log10 space."""
        while True:
            candidate = []
            for param_name in params_to_optimize:
                low  = parameter_ranges[param_name]["min"]
                high = parameter_ranges[param_name]["max"]
                logval = np.random.uniform(math.log10(low), math.log10(high))
                candidate.append(10.0 ** logval)

            if satisfies_constraints(candidate):
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

        If a mutation breaks the kcatLK constraint, only that gene is reverted; the
        rest of the individual is preserved.
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

            # Only kfLK/krLK can break the constraint; re-checking the other nine genes
            # is wasted work.
            if idx in (IDX_KFLK, IDX_KRLK) and not satisfies_constraints(candidate):
                candidate[idx] = old_val

        return (candidate,)

    def mate_and_validate(self, ind1, ind2):
        # 1. Save backups of the parents before DEAP modifies them in-place
        parent1_backup = ind1[:]
        parent2_backup = ind2[:]

        # 2. Perform the crossover
        tools.cxTwoPoint(ind1, ind2)

        # 3. Check constraints. If a child is invalid, revert it to its parent's state
        if not satisfies_constraints(ind1):
            ind1[:] = parent1_backup

        if not satisfies_constraints(ind2):
            ind2[:] = parent2_backup

        return ind1, ind2

    def _evaluate(self, individuals):
        """Solve the ODE for `individuals`, cache all four metrics, set scalar fitness."""
        if not individuals:
            return
        for ind, m in zip(individuals, self.toolbox.map(self.toolbox.evaluate, individuals)):
            ind.metrics = (float(m[0]), float(m[1]), float(m[2]), bool(m[3]))
            ind.fitness.values = (ind.metrics[0],)

    def _record(self, individuals, seen, points, fitnesses):
        """
        File every individual clearing the viability gate, returning the count of new
        ones. Membership is tested against a set of parameter tuples rather than the
        old `if ind not in all_viable_points` linear scan, which was O(n^2).
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
        self._evaluate(population)
        self.hof.update(population)
        new = self._record(population, seen, points, fitnesses)
        best = max(ind.metrics[0] for ind in population)
        print(f"  [restart {restart_idx}] gen   0 (initial) : best={best:.6f}  "
              f"new={new:5d}  total={len(points)}")

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
            self._evaluate(pending)

            self.hof.update(offspring)
            new = self._record(offspring, seen, points, fitnesses)
            population = self.toolbox.select(population + offspring, k=self.populationSize)

            best = max(ind.metrics[0] for ind in offspring)
            print(f"  [restart {restart_idx}] gen {gen:3d}           : best={best:.6f}  "
                  f"new={new:5d}  total={len(points)}  "
                  f"solved={len(pending)}/{len(offspring)}")

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
        """Expand a free-parameter vector into the full ODE parameter set for output."""
        kfLA, kfLK, krLK, kfLP, krLP, kfAP, DF, L, K, P, A = opt
        return {
            'Fitness':        fitness_val,
            'Period':         period_val,
            'Amplitude':      amplitude_val,
            # Strict oscillator verdict from calculate_oscillator_metrics: this carries
            # the damped (screen 3) and regularity (screen 4) tests, which the GA's own
            # period/amplitude gate does not apply. Recorded rather than enforced, so
            # filtering on it stays a post-processing choice.
            'Is_Oscillatory': bool(is_osc),
            # Free parameters
            'kfLA':   kfLA,
            'kfLK':   kfLK,
            'krLK':   krLK,
            'kfLP':   kfLP,
            'krLP':   krLP,
            'kfAP':   kfAP,
            'DF':     DF,
            'L':      L,
            'K':      K,
            'P':      P,
            'A':      A,
            # Derived / fixed (full ODE parameter set)
            'krLA':   KDLA * kfLA,
            'kcatLK': KMLK * kfLK - krLK,
            'kcatLP': KCAT_LP_FIXED,
            'kfAK':   KFAK_FIXED,
            'krAK':   KRAK_FIXED,
            'krAP':   KDAP * kfAP,
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
        try:
            viablePoints, viableFitness = self.findNominalValues()
        finally:
            # Close in a finally so a crash mid-run does not leak the worker pool.
            self.pool.close()
            self.pool.join()

        self.write_sortedParms(viablePoints, viableFitness)
        self.write_hallOfFame()
        print("GA complete.")
        return viablePoints, viableFitness


# ==================== MAIN ====================

if __name__ == "__main__":
    start_time = time.time()

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

    POP_SIZE   = 1_000_000
    N_GEN      = 100      # max generations within a single restart
    GEN_BUDGET = 100      # total generations across all restarts (the compute cap)
    RESTARTS   = 4        # max independent restarts within that budget
    PATIENCE   = 10       # abandon a basin after this many barren generations
    SAMPLING   = "random" # initial population sampling: "lhs" or "random"
    SEED       = None     # set an int for a reproducible run

    random_id = np.random.randint(1, 10000)
    outfile   = f"ga_oscillator_results_{random_id}.csv"
    print(f"Population size : {POP_SIZE}")
    print(f"Generations     : {N_GEN} max per restart, {GEN_BUDGET} total budget")
    print(f"Restarts        : up to {RESTARTS}, patience {PATIENCE} barren generations")
    print(f"Sampling method : {SAMPLING}")
    print(f"Seed            : {SEED}")
    print(f"Output file     : {outfile}")
    print(f"CPU cores       : {multiprocessing.cpu_count()}")
    print(f"Free parameters : {params_to_optimize}")
    print()

    solver = Solver(populationSize=POP_SIZE, NGEN=N_GEN, outfileName=outfile,
                    sampling=SAMPLING, restarts=RESTARTS, genBudget=GEN_BUDGET,
                    patience=PATIENCE, seed=SEED)
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
