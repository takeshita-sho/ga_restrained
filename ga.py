import os
import sys
from scipy.stats import qmc
# Force NumPy/SciPy to use only 1 thread per process to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pandas as pd
import time
import multiprocessing

from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
from oscillator_models.solve_ode import calc_fitness_math, full_model_ode, calculate_oscillator_metrics, process_single_row

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

KDLA = 20.0 / 7.0   # krLA   = KDLA * kfLA
KDAP = 29.0          # krAP   = KDAP * kfAP
KMLK  = 15.0          
KMLP     = 39.0          # krLP   = KMLP     * kfLP - KCAT_LP_FIXED

# Lower bound on kfLP so that krLP = 39*kfLP - 85.3 >= 1e-3
_KFLP_MIN = (KCAT_LP_FIXED + 1e-3) / KMLP   # ~2.187

parameter_ranges = {
    "kfLA": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "kfLK": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "krLK": {"min": 1e-3,      "max": 1e3},    # kr  [1e-3, 1e3]
    "kfLP": {"min": _KFLP_MIN, "max": 10.0},   # kf  (lower bound ensures krLP >= 1e-3)
    "kfAP": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "DF":   {"min": 1e2,      "max": 1e4},
    "L":    {"min": 1e-2,      "max": 1e2},
    "K":    {"min": 1e-3,      "max": 1e2},
    "P":    {"min": 1e-3,      "max": 1e1},
    "A":    {"min": 1e-2,      "max": 1e2},
}

params_to_optimize = ["kfLA", "kfLK", "krLK", "kfLP", "kfAP", "DF", "L", "K", "P", "A"]


def _build_ode_params(kfLA, kfLK, krLK, kfLP, kfAP, DF):
    """Assemble the 13-element ODE parameter vector from the free rate/DF params."""
    krLA   = KDLA * kfLA
    kcatLK = KMLK * kfLK - krLK
    krLP   = KMLP * kfLP - KCAT_LP_FIXED
    krAP   = KDAP * kfAP
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


    kcatLK = KMLK * kfLK - krLK
    krLP   = KMLP * kfLP - KCAT_LP_FIXED
    if kcatLK <= 0 or krLP <= 0:
        return (0.0, 0.0, 0.0)

    params = _build_ode_params(kfLA, kfLK, krLK, kfLP, kfAP, DF)
    params = np.concat([params, np.array([L, K, P, A], dtype=np.float64)])
    metrics = process_single_row(params, T=2000.0)

    return (metrics[0], metrics[1], metrics[2])

    # y0 = np.zeros(16)
    # y0[0], y0[1], y0[2], y0[3] = L, K, P, A

    # t_span = (0.0, 2000.0)
    # start_wall = time.time()

    # def timeout_event(t, y, *args):
    #     if time.time() - start_wall > 300.0:
    #         raise TimeoutError("ODE solver timed out after 10 minutes")
    #     return 1.0

    # try:
    #     sol = solve_ivp(
    #         fun=full_model_ode,
    #         t_span=t_span,
    #         y0=y0,
    #         args=(params,),
    #         method='Radau',
    #         rtol=1e-6,
    #         atol=1e-8,
    #         events=[timeout_event],
    #     )

    #     res = sol.y
    #     numerator   = res[5] + res[8] + res[9] + res[10] + res[11] + res[14] + res[15]
    #     denominator = numerator + res[3] + res[12] + res[13]

    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         Amem = np.where(denominator > 0, numerator / denominator, 0.0)

    #     metrics = calculate_oscillator_metrics(sol.t, Amem)
    #     fitness = metrics['Fitness']
    #     period = metrics['Period']
    #     amplitude = metrics['Amplitude']
    #     if fitness <= 0.0:
    #         # Only look at the second half of the time series to ignore initial transients
    #         halfway_idx = len(sol.t) // 2
    #         if halfway_idx > 10: # Ensure we have enough data points
    #             tail_Amem = Amem[halfway_idx:]
                
    #             # Calculate standard deviation. A higher std dev means the signal 
    #             # is fluctuating, which is a stepping stone to oscillation.
    #             signal_std = np.std(tail_Amem)
                
    #             # Give a microscopic fitness score based on standard deviation.
    #             # It won't beat a true oscillator, but it will beat a flatline.
    #             if signal_std > 1e-4:
    #                 # e.g., max partial credit of 0.1, scaling with variance
    #                 fitness = min(0.1, signal_std * 0.01) 
                    
    #                 # We can assign dummy period/amplitude since it's not a true oscillator yet
    #                 period = 0.0
    #                 amplitude = signal_std 

    #     return (fitness, period, amplitude)

    # except Exception:
    #     return (0.0, 0.0, 0.0)


# ==================== SOLVER CLASS ====================

class Solver:
    def __init__(self, populationSize=10000, NGEN=5, outfileName="ga_oscillator_results.csv", sampling="lhs"):
        self.populationSize = populationSize
        self.NGEN           = NGEN
        self.outfileName    = outfileName
        self.threshold      = 0.0   # keep solutions with fitness > 0
        self.indpb          = 0.2  # per-gene NON-mutation probability
        self.sampling       = sampling.lower()  # "lhs" or "random"

        # Guard against re-creating DEAP types if Solver is instantiated multiple times
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Candidate"):
            creator.create("Candidate", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("candidate",  self.generateCandidate)
        #self.toolbox.register("population", tools.initRepeat, list, self.toolbox.candidate)
        if self.sampling == "lhs":
            self.toolbox.register("population", self.generate_lhs_population)
        elif self.sampling == "random":
            self.toolbox.register("population", self.generate_random_population)
        else:
            raise ValueError(f"Unknown sampling method '{self.sampling}'; expected 'lhs' or 'random'.")
        self.toolbox.register("mutate",     self.mutateCandidate, indpb=self.indpb, mult=1.0)
        self.toolbox.register("select",     tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate",   evaluate_candidate)
        self.toolbox.register("mate", self.mate_and_validate)

        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.toolbox.register("map", self.pool.map)
    def generate_lhs_population(self, n):
        """Generates the initial population using LHS, respecting structural constraints."""
        print("Generating initial population via Latin Hypercube Sampling...")
        valid_candidates = []
        
        # Dimensions = number of free parameters
        dim = len(params_to_optimize)
        sampler = qmc.LatinHypercube(d=dim)
        
        # Extract log boundaries
        log_lower = [math.log10(parameter_ranges[p]["min"]) for p in params_to_optimize]
        log_upper = [math.log10(parameter_ranges[p]["max"]) for p in params_to_optimize]
        
        # Keep sampling in batches until we have 'n' valid candidates
        while len(valid_candidates) < n:
            # Over-sample to account for candidates that will fail the constraint
            batch_size = n * 2 
            sample = sampler.random(n=batch_size)
            
            # Scale samples to our log10 bounds
            scaled_log_sample = qmc.scale(sample, log_lower, log_upper)
            linear_sample = 10.0 ** scaled_log_sample
            
            for candidate in linear_sample:
                # 1 = kfLK, 2 = krLK based on params_to_optimize index
                kfLK = candidate[1]
                krLK = candidate[2]
                
                # Only keep if it passes the structural constraint
                if 15.0 * kfLK - krLK > 0:
                    # Convert to DEAP creator type
                    valid_candidates.append(creator.Candidate(candidate.tolist()))
                    
                    if len(valid_candidates) == n:
                        break
                        
        print(f"Successfully generated {n} LHS individuals.")
        return valid_candidates

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
            
            # Extract indices for kfLK (1) and krLK (2) based on params_to_optimize list
            kfLK = candidate[1]
            krLK = candidate[2]
            
            # Only return the candidate if it satisfies the structural constraint
            if 15.0 * kfLK - krLK > 0:
                return creator.Candidate(candidate)

    def mutateCandidate(self, candidate, indpb, mult):
        """
        Logarithmic mutation with per-gene constraint rejection.
        Mutates genes independently. If a mutation violates the biological constraints,
        that specific mutation is reverted, preserving the rest of the individual.
        """
        for idx in range(len(candidate)):
            if np.random.rand() < indpb:
                old_val = candidate[idx]
                
                # 1. Calculate new mutated value
                logval = math.log10(old_val)
                shift = np.random.uniform(-mult, mult)
                new_val = 10.0 ** (logval + shift)
                
                # 2. Clamp to predefined boundaries
                param_name = params_to_optimize[idx]
                min_val = parameter_ranges[param_name]["min"]
                max_val = parameter_ranges[param_name]["max"]
                candidate[idx] = max(min_val, min(new_val, max_val))
                
                # 3. Check structural constraints
                kfLK = candidate[1]
                krLK = candidate[2]
                
                # If the mutation broke the mathematical constraint, revert it
                if 15.0 * kfLK - krLK <= 0:
                    candidate[idx] = old_val
                    
        return (candidate,)

    def mate_and_validate(self, ind1, ind2):
        # 1. Save backups of the parents before DEAP modifies them in-place
        parent1_backup = ind1[:]
        parent2_backup = ind2[:]

        # 2. Perform the crossover
        tools.cxTwoPoint(ind1, ind2)

        # 3. Check constraints. If a child is invalid, revert it to its parent's state
        if 15.0 * ind1[1] - ind1[2] <= 0:
            ind1[:] = parent1_backup
            
        if 15.0 * ind2[1] - ind2[2] <= 0:
            ind2[:] = parent2_backup
            
        return ind1, ind2
    
    def findNominalValues(self):
        """Main GA loop. Returns all individuals whose fitness exceeded the threshold."""
        population = self.toolbox.population(n=self.populationSize)
        all_viable_points  = []
        all_viable_fitness = []

        print("BEGIN main GA loop.")
        for gen in range(self.NGEN):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.6, mutpb=0.4)
            fits = list(self.toolbox.map(self.toolbox.evaluate, offspring))

            for fit_val, ind in zip(fits, offspring):
                cost, period, amplitude = fit_val
                ind.fitness.values = (cost,)
                if cost > self.threshold and period >= 1.0 and amplitude >= 0.01:
                    if ind not in all_viable_points:
                        all_viable_points.append(ind[:])
                        all_viable_fitness.append([cost, period, amplitude])

            population = self.toolbox.select(population + offspring, k=self.populationSize)

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

            krLA   = KDLA * kfLA
            kcatLK = KMLK * kfLK - krLK
            krLP   = KMLP * kfLP - KCAT_LP_FIXED
            krAP   = KDAP * kfAP

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

    POP_SIZE = 1_000_000
    N_GEN    = 100
    SAMPLING = "random"   # initial population sampling: "lhs" or "random"

    random_id = np.random.randint(1, 10000)
    outfile   = f"ga_oscillator_results_{random_id}.csv"
    print(f"Population size : {POP_SIZE}")
    print(f"Generations     : {N_GEN}")
    print(f"Sampling method : {SAMPLING}")
    print(f"Output file     : {outfile}")
    print(f"CPU cores       : {multiprocessing.cpu_count()}")
    print(f"Free parameters : {params_to_optimize}")
    print()

    solver = Solver(populationSize=POP_SIZE, NGEN=N_GEN, outfileName=outfile, sampling=SAMPLING)
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
