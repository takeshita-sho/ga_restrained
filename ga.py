import os
# Force NumPy/SciPy to use only 1 thread per process to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import numpy as np
import pandas as pd
import time
import multiprocessing

from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
from solve_ode import calc_fitness_math, full_model_ode, calculate_oscillator_metrics

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==================== PARAMETER CONSTRAINTS ====================
#
# Free parameters (sampled by GA):
#   kfLA, kfLK, krLK, kfLP, kfAP, DF, L, K, P, A
#
# Fixed:
#   kfAK = 0.118,  krAK = 0.0609,  kcatLP = 85.3
#
# Derived (computed from free params, never directly sampled):
#   krLA   = (20/7) * kfLA
#   kcatLK = 15 * kfLK - krLK          (penalised to 0 if <= 0)
#   krLP   = 39 * kfLP - kcatLP        (kfLP lower bound ensures krLP >= 1e-3)
#   krAP   = 29 * kfAP

KFAK_FIXED    = 0.118
KRAK_FIXED    = 0.0609
KCAT_LP_FIXED = 85.3

KRLA_RATIO = 20.0 / 7.0   # krLA   = KRLA_RATIO * kfLA
KRAP_RATIO = 29.0          # krAP   = KRAP_RATIO * kfAP
KCAT_LK_A  = 15.0          # kcatLK = KCAT_LK_A  * kfLK - krLK
KRLP_A     = 39.0          # krLP   = KRLP_A     * kfLP - KCAT_LP_FIXED

# Lower bound on kfLP so that krLP = 39*kfLP - 85.3 >= 1e-3
_KFLP_MIN = (KCAT_LP_FIXED + 1e-3) / KRLP_A   # ~2.187

parameter_ranges = {
    "kfLA": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "kfLK": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "krLK": {"min": 1e-3,      "max": 1e3},    # kr  [1e-3, 1e3]
    "kfLP": {"min": _KFLP_MIN, "max": 10.0},   # kf  (lower bound ensures krLP >= 1e-3)
    "kfAP": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "DF":   {"min": 1e-1,      "max": 1e3},
    "L":    {"min": 1e-1,      "max": 1e2},
    "K":    {"min": 1e-3,      "max": 1e2},
    "P":    {"min": 1e-3,      "max": 1e2},
    "A":    {"min": 1e-3,      "max": 1e2},
}

params_to_optimize = ["kfLA", "kfLK", "krLK", "kfLP", "kfAP", "DF", "L", "K", "P", "A"]


def _build_ode_params(kfLA, kfLK, krLK, kfLP, kfAP, DF):
    """Assemble the 13-element ODE parameter vector from the free rate/DF params."""
    krLA   = KRLA_RATIO * kfLA
    kcatLK = KCAT_LK_A * kfLK - krLK
    krLP   = KRLP_A * kfLP - KCAT_LP_FIXED
    krAP   = KRAP_RATIO * kfAP
    return np.array([kfLA, krLA, kfLK, krLK, kcatLK,
                     kfLP, krLP, KCAT_LP_FIXED,
                     KFAK_FIXED, KRAK_FIXED,
                     kfAP, krAP, DF], dtype=np.float64)


# ==================== FITNESS FUNCTION ====================

def evaluate_candidate(candidate):
    """
    Top-level fitness function (module-level for multiprocessing pickling).
    Returns a 1-tuple (fitness,) as required by DEAP FitnessMax.
    Candidates with kcatLK <= 0 are penalised with fitness 0.
    """
    kfLA, kfLK, krLK, kfLP, kfAP, DF, L, K, P, A = candidate

    # Validate derived parameters
    kcatLK = KCAT_LK_A * kfLK - krLK
    krLP   = KRLP_A * kfLP - KCAT_LP_FIXED
    if kcatLK <= 0 or krLP <= 0:
        return (0.0, 0.0, 0.0)

    params = _build_ode_params(kfLA, kfLK, krLK, kfLP, kfAP, DF)
    y0 = np.zeros(16)
    y0[0], y0[1], y0[2], y0[3] = L, K, P, A

    t_span = (0.0, 12000.0)
    start_wall = time.time()

    def timeout_event(t, y, *args):
        if time.time() - start_wall > 600.0:
            raise TimeoutError("ODE solver timed out after 10 minutes")
        return 1.0

    try:
        sol = solve_ivp(
            fun=full_model_ode,
            t_span=t_span,
            y0=y0,
            args=(params,),
            method='Radau',
            rtol=1e-6,
            atol=1e-8,
            events=[timeout_event],
        )

        res = sol.y
        numerator   = res[5] + res[8] + res[9] + res[10] + res[11] + res[14] + res[15]
        denominator = numerator + res[3] + res[12] + res[13]

        with np.errstate(divide='ignore', invalid='ignore'):
            Amem = np.where(denominator > 0, numerator / denominator, 0.0)

        metrics = calculate_oscillator_metrics(sol.t, Amem)
        return (metrics['Fitness'], metrics['Period'], metrics['Amplitude'])

    except Exception:
        return (0.0, 0.0, 0.0)


# ==================== SOLVER CLASS ====================

class Solver:
    def __init__(self, populationSize=10000, NGEN=5, outfileName="ga_oscillator_results.csv"):
        self.populationSize = populationSize
        self.NGEN           = NGEN
        self.outfileName    = outfileName
        self.threshold      = 0.0   # keep solutions with fitness > 0
        self.indpb          = 0.75  # per-gene NON-mutation probability

        # Guard against re-creating DEAP types if Solver is instantiated multiple times
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Candidate"):
            creator.create("Candidate", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("candidate",  self.generateCandidate)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.candidate)
        self.toolbox.register("mate",       tools.cxTwoPoint)
        self.toolbox.register("mutate",     self.mutateCandidate, indpb=self.indpb, mult=0.5)
        self.toolbox.register("select",     tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate",   evaluate_candidate)

        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.toolbox.register("map", self.pool.map)

    def generateCandidate(self):
        """Sample each free parameter uniformly in log10 space."""
        candidate = []
        for param_name in params_to_optimize:
            low  = parameter_ranges[param_name]["min"]
            high = parameter_ranges[param_name]["max"]
            logval = np.random.uniform(math.log10(low), math.log10(high))
            candidate.append(10.0 ** logval)
        return creator.Candidate(candidate)

    def mutateCandidate(self, candidate, indpb, mult):
        """
        Mutate each gene with probability (1 - indpb) by a multiplicative factor
        drawn uniformly from [1 - mult, 1 + mult].  Result is clamped to [min, max].
        """
        for idx, val in enumerate(candidate):
            if np.random.rand() >= indpb:
                scale = np.random.uniform(1.0 - mult, 1.0 + mult)
                candidate[idx] = val * scale
                param_name = params_to_optimize[idx]
                min_val = parameter_ranges[param_name]["min"]
                max_val = parameter_ranges[param_name]["max"]
                candidate[idx] = max(min_val, min(candidate[idx], max_val))
        return (candidate,)

    def findNominalValues(self):
        """Main GA loop. Returns all individuals whose fitness exceeded the threshold."""
        population = self.toolbox.population(n=self.populationSize)
        all_viable_points  = []
        all_viable_fitness = []

        print("BEGIN main GA loop.")
        for gen in range(self.NGEN):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.8, mutpb=0.2)
            fits = list(self.toolbox.map(self.toolbox.evaluate, offspring))

            for fit_val, ind in zip(fits, offspring):
                cost, period, amplitude = fit_val
                ind.fitness.values = (cost,)
                if cost > self.threshold:
                    if ind not in all_viable_points:
                        all_viable_points.append(ind[:])
                        all_viable_fitness.append([cost, period, amplitude])

            population = self.toolbox.select(offspring, k=len(population))

            best_this_gen = max(f[0] for f in fits) if fits else 0.0
            print(f"=== Generation {gen} ===")
            print(f"  Best fitness this generation : {best_this_gen:.6f}")
            print(f"  Viable solutions accumulated : {len(all_viable_points)}")

        return all_viable_points, all_viable_fitness

    def write_sortedParms(self, viablePoints, viableFitness):
        """Sort viable solutions by descending fitness and write full parameter set to CSV."""
        fit_values = [vf[0] for vf in viableFitness]
        idx_sorted = np.argsort([-v for v in fit_values])

        rows = []
        for rank, idx in enumerate(idx_sorted):
            fitness_val, period_val, amplitude_val = viableFitness[idx]
            opt = viablePoints[idx]
            kfLA, kfLK, krLK, kfLP, kfAP, DF, L, K, P, A = opt

            krLA   = KRLA_RATIO * kfLA
            kcatLK = KCAT_LK_A * kfLK - krLK
            krLP   = KRLP_A * kfLP - KCAT_LP_FIXED
            krAP   = KRAP_RATIO * kfAP

            rows.append({
                'Rank':      rank,
                'Fitness':   fitness_val,
                'Period':    period_val,
                'Amplitude': amplitude_val,
                # Free parameters
                'kfLA':   kfLA,
                'kfLK':   kfLK,
                'krLK':   krLK,
                'kfLP':   kfLP,
                'kfAP':   kfAP,
                'DF':     DF,
                'L':      L,
                'K':      K,
                'P':      P,
                'A':      A,
                # Derived / fixed (full ODE parameter set)
                'krLA':   krLA,
                'kcatLK': kcatLK,
                'krLP':   krLP,
                'kcatLP': KCAT_LP_FIXED,
                'kfAK':   KFAK_FIXED,
                'krAK':   KRAK_FIXED,
                'krAP':   krAP,
            })

        pd.DataFrame(rows).to_csv(self.outfileName, index=False)
        print(f"Wrote {len(rows)} viable solutions to {self.outfileName}")

    def run(self):
        """Run the full GA pipeline and write results."""
        viablePoints, viableFitness = self.findNominalValues()
        self.write_sortedParms(viablePoints, viableFitness)
        self.pool.close()
        self.pool.join()
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

    POP_SIZE = 10000
    N_GEN    = 5

    random_id = np.random.randint(1, 10000)
    outfile   = f"ga_oscillator_results_{random_id}.csv"
    print(f"Population size : {POP_SIZE}")
    print(f"Generations     : {N_GEN}")
    print(f"Output file     : {outfile}")
    print(f"CPU cores       : {multiprocessing.cpu_count()}")
    print(f"Free parameters : {params_to_optimize}")
    print()

    solver = Solver(populationSize=POP_SIZE, NGEN=N_GEN, outfileName=outfile)
    viablePoints, viableFitness = solver.run()

    if viableFitness:
        best_fit = max(viableFitness, key=lambda x: x[0])
        best_idx = viableFitness.index(best_fit)
        print(f"\nBest fitness : {best_fit[0]:.6f}")
        print(f"Best params  : {dict(zip(params_to_optimize, viablePoints[best_idx]))}")
    else:
        print("\nNo viable solutions found (all fitness <= 0).")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.2f} minutes")
