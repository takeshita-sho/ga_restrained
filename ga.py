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
    "kfLA": {"min": 1e-4,      "max": 10.0},   # kf  [1e-3, 10]
    "kfLK": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "krLK": {"min": 1e-3,      "max": 1e3},    # kr  [1e-3, 1e3]
    "kfLP": {"min": _KFLP_MIN, "max": 10.0},   # kf  (lower bound ensures krLP >= 1e-3)
    "kfAP": {"min": 1e-3,      "max": 10.0},   # kf  [1e-3, 10]
    "DF":   {"min": 1e2,      "max": 1e4},
    "L":    {"min": 1e-1,      "max": 1e2},
    "K":    {"min": 1e-3,      "max": 1e2},
    "P":    {"min": 1e-3,      "max": 1e2},
    "A":    {"min": 1e-3,      "max": 1e2},
}

params_to_optimize = ["kfLA", "kfLK", "krLK", "kfLP", "kfAP", "DF", "L", "K", "P", "A"]

# Bounds for tools.mutPolynomialBounded, expressed in log10 space so that mutation stays
# multiplicative (matching the log-uniform sampling used to build the population).
# These MUST be plain Python lists: DEAP tests isinstance(low, Sequence), and a numpy
# array fails that test, so it would be repeat()-ed as a whole-array "scalar" bound and
# silently broadcast over every gene without raising.
LOG_LOW = [math.log10(parameter_ranges[p]["min"]) for p in params_to_optimize]
LOG_UP  = [math.log10(parameter_ranges[p]["max"]) for p in params_to_optimize]

LOG_SPAN = [u - l for l, u in zip(LOG_LOW, LOG_UP)]


def _reflect(x, lo, hi):
    """Fold x back into [lo, hi] by mirroring off the bounds instead of clamping.

    Clamping maps every overshoot onto the bound itself, which is what was building up
    density at the two ends of kfLP. Reflection redistributes it into the interior.
    """
    span = hi - lo
    if span <= 0.0:
        return lo
    y = (x - lo) % (2.0 * span)          # float % is always >= 0 for a positive divisor
    return lo + (2.0 * span - y if y > span else y)


# Gene indices participating in the kcatLK = KMLK*kfLK - krLK > 0 constraint
KFLK_IDX = params_to_optimize.index("kfLK")   # 1
KRLK_IDX = params_to_optimize.index("krLK")   # 2


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

    

# ==================== SOLVER CLASS ====================

class Solver:
    def __init__(self, populationSize=10000, NGEN=5, outfileName="ga_oscillator_results.csv", sampling="lhs",
                 cx_indpb=0.5, mutation="polynomial", mut_eta=20.0, mut_delta_scale=0.25):
        self.populationSize = populationSize
        self.NGEN           = NGEN
        self.outfileName    = outfileName
        self.threshold      = 0.0   # keep solutions with fitness > 0
        self.indpb          = 0.75  # per-gene mutation probability
        self.cx_eta         = 2.0   # SBX crowding degree (distribution index)
        self.cx_indpb       = cx_indpb  # per-gene crossover probability
        self.sampling       = sampling.lower()  # "lhs" or "random"
        self.mutation       = mutation.lower()  # "polynomial" or "log_uniform"
        self.mut_eta        = mut_eta   # polynomial mutation crowding degree (high = small steps)
        self.n_mut_reverts  = 0     # count of mutations rolled back by the kcatLK constraint
        self.mut_delta_scale = mut_delta_scale

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
        if self.mutation == "plm":
            self.toolbox.register("mutate", self.mutate_plm,
                                  indpb=self.indpb, eta=self.mut_eta,
                                  delta_scale=self.mut_delta_scale)
        elif self.mutation == "polynomial":
            self.toolbox.register("mutate", self.mutate_polynomial,
                                  indpb=self.indpb, eta=self.mut_eta)
        elif self.mutation == "log_uniform":
            self.toolbox.register("mutate", self.mutateCandidate,
                                  indpb=self.indpb, mult=1.0)
        else:
            raise ValueError(f"Unknown mutation method '{self.mutation}'; "
                             f"expected 'polynomial' or 'log_uniform'.")
        # tournsize must be an int: DEAP's selRandom does range(tournsize)
        self.toolbox.register("select",     self.select_tournament_fast,
                              tournsize=max(2, int(0.05 * self.populationSize)))
        self.toolbox.register("evaluate",   evaluate_candidate)
        self.toolbox.register("mate", self.mate_and_validate,
                              eta=self.cx_eta, indpb=self.cx_indpb)

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
                kfLK = candidate[KFLK_IDX]
                krLK = candidate[KRLK_IDX]

                # Only keep if it passes the structural constraint
                if KMLK * kfLK - krLK > 0:
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
            
            kfLK = candidate[KFLK_IDX]
            krLK = candidate[KRLK_IDX]

            # Only return the candidate if it satisfies the structural constraint
            if KMLK * kfLK - krLK > 0:
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
                kfLK = candidate[KFLK_IDX]
                krLK = candidate[KRLK_IDX]
                
                # If the mutation broke the mathematical constraint, revert it
                if KMLK * kfLK - krLK <= 0:
                    candidate[idx] = old_val

        return (candidate,)
    @staticmethod
    def _plm_delta(eta):
        """Polynomial mutation factor delta_i from SI Eq. 9a/9b."""
        u = np.random.rand()
        p = 1.0 / (eta + 1.0)
        if u <= 0.5:
            return (2.0 * u) ** p - 1.0
        return 1.0 - (2.0 * (1.0 - u)) ** p

    def mutate_plm(self, candidate, indpb, eta, delta_scale=1.0):
        """
        Paper-faithful Polynomial Mutation (SI SIII.1): c'_i = c_i + Delta * delta_i,
        with Delta = 1.0 and eta = 2, applied to log10(gene) so a step of Delta is one
        decade. Unlike DEAP's mutPolynomialBounded the step is NOT scaled by (up - low),
        so krLK over [1e-3, 1e3] gets the same ~0.2-decade median step as every other gene
        instead of a 1.05-decade one.
        """
        backup = candidate[:]
        for idx in range(len(candidate)):
            if np.random.rand() < indpb:
                step = delta_scale * LOG_SPAN[idx] * self._plm_delta(eta)
                new_log = _reflect(math.log10(candidate[idx]) + step,
                                   LOG_LOW[idx], LOG_UP[idx])
                candidate[idx] = 10.0 ** new_log

        # 10 ** log10(x) can land a hair outside the linear bound.
        for idx, param_name in enumerate(params_to_optimize):
            min_val = parameter_ranges[param_name]["min"]
            max_val = parameter_ranges[param_name]["max"]
            candidate[idx] = max(min_val, min(candidate[idx], max_val))

        if KMLK * candidate[KFLK_IDX] - candidate[KRLK_IDX] <= 0:
            candidate[KFLK_IDX] = backup[KFLK_IDX]
            candidate[KRLK_IDX] = backup[KRLK_IDX]
            self.n_mut_reverts += 1

        return (candidate,)

    def mutate_polynomial(self, candidate, indpb, eta):
        """
        NSGA-II polynomial mutation (DEAP tools.mutPolynomialBounded) applied in log10
        space, with the kcatLK constraint enforced by reverting the (kfLK, krLK) pair.

        Box bounds are enforced by the operator itself, which clamps every mutated gene
        into [low, up]; kfLP's lower bound therefore keeps krLP = KMLP*kfLP - kcatLP > 0
        for free. The only constraint left is kcatLK = KMLK*kfLK - krLK > 0, which couples
        two genes and so cannot be expressed as a per-gene bound. If a mutation breaks it,
        only those two genes revert to their parent values -- which were feasible, so the
        result always is too -- and the other eight genes keep their mutations.

        Mutating log10(gene) rather than the gene keeps steps multiplicative, matching the
        log-uniform sampling in generateCandidate/generate_lhs_population. In linear space
        an eta=20 step is a few percent of the range, i.e. O(10) absolute on krLK over
        [1e-3, 1e3], which would leave the low decades unreachable.
        """
        backup = candidate[:]

        log_genes = [math.log10(v) for v in candidate]
        tools.mutPolynomialBounded(log_genes, eta=eta, low=LOG_LOW, up=LOG_UP,
                                   indpb=indpb)

        # Re-clamp on the way back: 10 ** log10(x) can land a hair outside the bound.
        for idx, log_val in enumerate(log_genes):
            param_name = params_to_optimize[idx]
            min_val = parameter_ranges[param_name]["min"]
            max_val = parameter_ranges[param_name]["max"]
            candidate[idx] = max(min_val, min(10.0 ** log_val, max_val))

        if KMLK * candidate[KFLK_IDX] - candidate[KRLK_IDX] <= 0:
            candidate[KFLK_IDX] = backup[KFLK_IDX]
            candidate[KRLK_IDX] = backup[KRLK_IDX]
            self.n_mut_reverts += 1

        return (candidate,)

    def mate_and_validate(self, ind1, ind2, eta=2.0, indpb=0.5):
        """
        Simulated binary crossover with a per-gene crossover probability.

        DEAP's cxSimulatedBinary takes only eta (crowding degree) and blends every gene
        unconditionally, so the per-gene probability is applied here: genes not selected
        revert to the parent value. Selected genes are clamped to their parameter_ranges
        bounds -- SBX extrapolates outside the parents' interval and can otherwise drive a
        gene negative, which would make math.log10 in mutateCandidate raise ValueError.
        """
        # 1. Save backups of the parents before DEAP modifies them in-place
        parent1_backup = ind1[:]
        parent2_backup = ind2[:]

        # 2. Perform the crossover
        log1 = [math.log10(v) for v in ind1]
        log2 = [math.log10(v) for v in ind2]
        tools.cxSimulatedBinary(log1, log2, eta)

        # 3. Keep only the genes selected with probability indpb, clamping them to bounds
        for idx in range(len(ind1)):
            if np.random.rand() < indpb:
                min_val = parameter_ranges[params_to_optimize[idx]]["min"]
                max_val = parameter_ranges[params_to_optimize[idx]]["max"]
                ind1[idx] = max(min_val, min(
                    10.0 ** _reflect(log1[idx], LOG_LOW[idx], LOG_UP[idx]), max_val))
                ind2[idx] = max(min_val, min(
                    10.0 ** _reflect(log2[idx], LOG_LOW[idx], LOG_UP[idx]), max_val))
            else:
                ind1[idx] = parent1_backup[idx]
                ind2[idx] = parent2_backup[idx]

        # 4. Check constraints. If a child is invalid, revert it to its parent's state
        if KMLK * ind1[KFLK_IDX] - ind1[KRLK_IDX] <= 0:
            ind1[:] = parent1_backup

        if KMLK * ind2[KFLK_IDX] - ind2[KRLK_IDX] <= 0:
            ind2[:] = parent2_backup
            
        return ind1, ind2
    def select_tournament_fast(self, individuals, k, tournsize, chunk=2048):
        """
        Vectorised replacement for tools.selTournament. Identical semantics -- k tournaments
        of `tournsize` individuals drawn with replacement, winner = highest fitness, ties to
        the lowest index -- but the k x tournsize sampling and the argmax run in numpy.

        Chunked over k because the full index matrix at k=100,000 and tournsize=5,000 is
        5e8 entries (~2 GB as int32, plus the same again for the gathered fitnesses).
        Returns references, not copies; varAnd clones before mutating, as DEAP expects.
        """
        n = len(individuals)
        fitness = np.fromiter((ind.fitness.values[0] for ind in individuals),
                              dtype=np.float64, count=n)
        winners = np.empty(k, dtype=np.int64)
        for start in range(0, k, chunk):
            rows = min(chunk, k - start)
            idx = np.random.randint(0, n, size=(rows, tournsize), dtype=np.int32)
            winners[start:start + rows] = idx[np.arange(rows), np.argmax(fitness[idx], axis=1)]
        return [individuals[i] for i in winners]

    def _score_and_record(self, individuals, viable_points, viable_fitness, viable_seen):
        """
        Evaluate only individuals whose fitness is stale, then record the viable ones.

        varAnd deletes fitness values on every individual it crosses over or mutates, so
        `not ind.fitness.valid` is exactly the set that changed; untouched clones keep their
        parent's score and are skipped, which also stops them being recorded twice.
        Membership is tested against a set of tuples rather than `ind not in list`, which was
        a linear scan of a list that grows to O(10^5).
        """
        pending = [ind for ind in individuals if not ind.fitness.valid]
        if not pending:
            return 0

        for (cost, period, amplitude), ind in zip(
                self.toolbox.map(self.toolbox.evaluate, pending), pending):
            ind.fitness.values = (cost,)
            if cost > self.threshold and period >= 1.0 and amplitude >= 0.01:
                key = tuple(ind)
                if key not in viable_seen:
                    viable_seen.add(key)
                    viable_points.append(ind[:])
                    viable_fitness.append([cost, period, amplitude])
        return len(pending)
    def findNominalValues(self):
        """Main GA loop. Returns all individuals whose fitness exceeded the threshold."""
        population = self.toolbox.population(n=self.populationSize)
        all_viable_points  = []
        all_viable_fitness = []
        all_viable_seen    = set()

        n_eval = self._score_and_record(population, all_viable_points,
                                        all_viable_fitness, all_viable_seen)
        print(f"=== Generation 0 (initial population) ===")
        print(f"  Evaluated                    : {n_eval}")
        print(f"  Best fitness                 : "
              f"{max(ind.fitness.values[0] for ind in population):.6f}")
        print(f"  Viable solutions accumulated : {len(all_viable_points)}")

        print("BEGIN main GA loop.")
        for gen in range(1, self.NGEN + 1):
            reverts_before = self.n_mut_reverts
            mating_pool = self.toolbox.select(population, k=self.populationSize)
            offspring   = algorithms.varAnd(mating_pool, self.toolbox, cxpb=0.75, mutpb=0.95)
            self._score_and_record(offspring, all_viable_points,
                                   all_viable_fitness, all_viable_seen)
            population = offspring

            best_this_gen = max(ind.fitness.values[0] for ind in population)
            print(f"=== Generation {gen} ===")
            print(f"  Best fitness this generation : {best_this_gen:.6f}")
            print(f"  Viable solutions accumulated : {len(all_viable_points)}")
            print(f"  kcatLK mutation reverts      : "
                  f"{self.n_mut_reverts - reverts_before} / {len(offspring)}")

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

    POP_SIZE = 100_000
    N_GEN    = 100
    SAMPLING = "random"   # initial population sampling: "lhs" or "random"
    CX_INDPB = 0.3        # per-gene crossover probability (SBX)
    MUTATION = "plm"  # mutation operator: "polynomial" or "log_uniform"
    MUT_ETA  = 2.0          # polynomial mutation crowding degree (high = small steps)
    MUT_DELTA_SCALE = 0.25

    random_id = np.random.randint(1, 10000)
    outfile   = f"ga_oscillator_results_{random_id}.csv"
    print(f"Population size : {POP_SIZE}")
    print(f"Generations     : {N_GEN}")
    print(f"Sampling method : {SAMPLING}")
    print(f"Crossover indpb : {CX_INDPB}")
    print(f"Mutation op     : {MUTATION}" + (f" (eta={MUT_ETA})" if MUTATION in ("polynomial","plm") else ""))
    print(f"Output file     : {outfile}")
    print(f"CPU cores       : {multiprocessing.cpu_count()}")
    print(f"Free parameters : {params_to_optimize}")
    print(f"Mutation delta scale : {MUT_DELTA_SCALE}")
    print()

    solver = Solver(populationSize=POP_SIZE, NGEN=N_GEN, outfileName=outfile,
                    sampling=SAMPLING, cx_indpb=CX_INDPB, mutation=MUTATION,
                    mut_eta=MUT_ETA, mut_delta_scale=MUT_DELTA_SCALE)
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
