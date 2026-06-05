import os
# Force NumPy/SciPy to use only 1 thread per process to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import sys
from scipy.signal import find_peaks
from scipy.fft import rfft
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm # pip install tqdm for the progress bar
import multiprocessing as mp

INCLUDE_CONDITIONED_OUTPUT = False
# --- 1. NUMBA OPTIMIZED MATH FUNCTIONS ---
# Added cache=True so the 48 workers don't recompile simultaneously
@njit(fastmath=True, cache=True)
def calc_fitness_math(spec_peaks, Y):
    sharpness_sum = 0.0
    gamma_sum = 0.0
    num_peaks = len(spec_peaks)
    if num_peaks == 0:
        return 0.0
        
    peak_amplitudes = np.zeros(num_peaks)
    for i in range(num_peaks):
        idx = spec_peaks[i]
        if idx > 0 and idx < len(Y) - 1:
            y0, y1, y2 = Y[idx-1], Y[idx], Y[idx+1]
            mean = (y0 + y1 + y2) / 3.0
            variance = ((y0 - mean)**2 + (y1 - mean)**2 + (y2 - mean)**2) / 3.0
            sharpness_sum += np.sqrt(variance)
            peak_amplitudes[i] = Y[idx]
            
    if num_peaks >= 2:
        for i in range(1, num_peaks):
            gamma_sum += np.abs(peak_amplitudes[i] - peak_amplitudes[i-1])
            
    return sharpness_sum + gamma_sum

@njit(fastmath=True, cache=True)
def full_model_ode(t, y, params):
    L, K, P, A, Lp, LpA, LK, LpP, LpAK, LpAP, LpAKL, LpAPLp, AK, AP, AKL, APLp = y
    kfLA, krLA, kfLK, krLK, kcatLK, kfLP, krLP, kcatLP, kfAK, krAK, kfAP, krAP, DF = params
      
    Lp_A_flux = kfLA * Lp * A - krLA * LpA
    Lp_AK_flux = kfLA * Lp * AK - krLA * LpAK
    Lp_AP_flux = kfLA * Lp * AP - krLA * LpAP
    LpA_K_flux = kfAK * LpA * K - krAK * LpAK
    LpA_P_flux = kfAP * LpA * P - krAP * LpAP
    
    v_L_K = kfLK * L * K - krLK * LK
    v_cat_LK = kcatLK * LK
    v_Lp_P = kfLP * Lp * P - krLP * LpP
    v_cat_LpP = kcatLP * LpP
    
    v_LpAK_L = (kfLK * DF) * LpAK * L - krLK * LpAKL
    v_cat_LpAKL = kcatLK * LpAKL
    v_Lp_LpAP = (kfLP * DF) * Lp * LpAP - krLP * LpAPLp
    v_cat_LpAPLp = kcatLP * LpAPLp
    
    v_Lp_AKL = (kfLA * DF) * Lp * AKL - krLA * LpAKL
    v_Lp_APLp = (kfLA * DF) * Lp * APLp - krLA * LpAPLp
    v_A_K = kfAK * A * K - krAK * AK
    v_A_P = kfAP * A * P - krAP * AP
    v_A_LK = kfAK * A * LK - krAK * AKL
    v_A_LpP = kfAP * A * LpP - krAP * APLp
    v_LpA_LK = (kfAK * DF) * LpA * LK - krAK * LpAKL
    v_LpA_LpP = (kfAP * DF) * LpA * LpP - krAP * LpAPLp
    v_AK_L = kfLK * AK * L - krLK * AKL
    v_cat_AKL = kcatLK * AKL
    v_AP_Lp = kfLP * AP * Lp - krLP * APLp
    v_cat_APLp = kcatLP * APLp

    dL = -v_L_K + v_cat_LpP - v_LpAK_L + v_cat_LpAPLp - v_AK_L + v_cat_APLp
    dK = -LpA_K_flux - v_L_K + v_cat_LK - v_A_K
    dP = -LpA_P_flux - v_Lp_P + v_cat_LpP - v_A_P
    dA = -Lp_A_flux - v_A_K - v_A_P - v_A_LK - v_A_LpP
    dLp = (-Lp_A_flux + v_cat_LK - v_Lp_P - v_Lp_LpAP + v_cat_LpAKL 
           - Lp_AK_flux - v_Lp_AKL - Lp_AP_flux - v_Lp_APLp + v_cat_AKL - v_AP_Lp)
    dLpA = Lp_A_flux - LpA_K_flux - LpA_P_flux - v_LpA_LK - v_LpA_LpP
    dLK = v_L_K - v_cat_LK - v_A_LK - v_LpA_LK
    dLpP = v_Lp_P - v_cat_LpP - v_A_LpP - v_LpA_LpP
    dLpAK = LpA_K_flux - v_LpAK_L + v_cat_LpAKL + Lp_AK_flux
    dLpAP = LpA_P_flux - v_Lp_LpAP + v_cat_LpAPLp + Lp_AP_flux
    dLpAKL = v_LpAK_L - v_cat_LpAKL + v_Lp_AKL + v_LpA_LK
    dLpAPLp = v_Lp_LpAP - v_cat_LpAPLp + v_Lp_APLp + v_LpA_LpP
    dAK = v_A_K - Lp_AK_flux - v_AK_L + v_cat_AKL
    dAP = v_A_P - Lp_AP_flux - v_AP_Lp + v_cat_APLp
    dAKL = v_A_LK - v_Lp_AKL + v_AK_L - v_cat_AKL
    dAPLp = v_A_LpP - v_Lp_APLp + v_AP_Lp - v_cat_APLp

    return np.array([dL, dK, dP, dA, dLp, dLpA, dLK, dLpP, dLpAK, dLpAP, dLpAKL, dLpAPLp, dAK, dAP, dAKL, dAPLp])

# --- 2. ANALYSIS FUNCTIONS ---
def calculate_oscillator_metrics(time_arr, amem):
    cutoff_index = int(0.1 * len(time_arr))
    t_sliced = time_arr[cutoff_index:]
    a_sliced = amem[cutoff_index:]
    
    if len(t_sliced) < 3:
        return {'Fitness': 0, 'Period': 0, 'Amplitude': 0, 'Is_Oscillatory': False}

    peaks, _ = find_peaks(a_sliced, prominence=0.01)
    troughs, _ = find_peaks(-a_sliced, prominence=0.01)

    period, amplitude = 0.0, 0.0
    is_oscillatory = False

    if len(peaks) >= 2 and len(troughs) >= 1:
        last_peak_idx = peaks[-1]
        prev_peak_idx = peaks[-2]
        period = t_sliced[last_peak_idx] - t_sliced[prev_peak_idx]
        
        idx_closest = np.argmin(np.abs(troughs - last_peak_idx))
        closest_trough_idx = troughs[idx_closest]
        amplitude = a_sliced[last_peak_idx] - a_sliced[closest_trough_idx]

        if amplitude > 0.01:
            is_oscillatory = True

    N = len(a_sliced)
    yf = rfft(a_sliced)
    Y = np.abs(yf) / (N / 2)
    spec_peaks, _ = find_peaks(Y, height=1e-2)
    fitness = calc_fitness_math(spec_peaks, Y)

    return {"Fitness": fitness, "Period": period, "Amplitude": amplitude, "Is_Oscillatory": is_oscillatory}

def process_single_row(row_vector):
    t_span = (0.0, 12000.0)
    L_init, K_init, P_init, A_init = row_vector[13], row_vector[14], row_vector[15], row_vector[16]
    y0 = np.zeros(16)
    y0[0:4] = [L_init, K_init, P_init, A_init]
    params = np.array(row_vector[0:13], dtype=np.float64)
    
    # Real-Time Safety Timeout (kills the solver if it gets stuck on a stiff ODE for > 600 seconds)
    start_time_real = time.time()
    def timeout_event(t, y, *args):
        if time.time() - start_time_real > 600.0:
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
            events=[timeout_event]
        )
        
        res = sol.y
        numerator = (res[5] + res[8] + res[9] + res[10] + res[11] + res[14] + res[15])
        denominator = numerator + res[3] + res[12] + res[13]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Amem = np.where(denominator > 0, numerator / denominator, 0)
            
        metrics = calculate_oscillator_metrics(sol.t, Amem)
        return np.array([
                            metrics['Fitness'],
                            metrics['Period'],
                            metrics['Amplitude'],
                            metrics['Is_Oscillatory'],
                            *row_vector
                        ], dtype=object)

    except Exception:
        result = {'Fitness': 0, 'Period': 0, 'Amplitude': 0, 'Is_Oscillatory': False}
        
        return np.array([
                    result['Fitness'],
                    result['Period'],
                    result['Amplitude'],
                    result['Is_Oscillatory'],
                    *row_vector
                ], dtype=object)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_csv> <output_csv> [condition_csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    if len(sys.argv) > 3:
        INCLUDE_CONDITIONED_OUTPUT = True
        condition_file = sys.argv[2]
        output_file = sys.argv[3]
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        df_condition = pd.read_csv(condition_file,usecols=['Period', 'Amplitude'])
        df['Conditioned_Period'] = df_condition['Period']
        df['Conditioned_Amplitude'] = df_condition['Amplitude']
        del df_condition
    else:
        output_file = sys.argv[2]
        df = pd.read_csv(input_file)
        cols_to_move = ['Period', 'Amplitude']
        if all(c in df.columns for c in cols_to_move):
            other_cols = [c for c in df.columns if c not in cols_to_move]
            df = df[other_cols + cols_to_move]
            df = df.rename(columns={'Period': 'Conditioned_Period', 'Amplitude': 'Conditioned_Amplitude'})
    
    
    
    data_matrix = df.to_numpy(dtype=np.float64)
    total_rows = len(data_matrix)
    
    headers = ['Fitness', 'Period', 'Amplitude', 'Is_Oscillatory'] + list(df.columns)
    pd.DataFrame(columns=headers).to_csv(output_file, index=False)
    
    max_workers = os.cpu_count()
    print(f"Starting parallel simulation across {max_workers} physical cores...")
    
    # ==========================================
    # NUMBA DUMMY RUN (Prevents the 48-core lockup)
    # ==========================================
    print("Pre-compiling Numba functions to prevent deadlock...")
    dummy_Y = np.random.rand(100)
    dummy_peaks = np.array([10, 20, 30])
    _ = calc_fitness_math(dummy_peaks, dummy_Y)
    
    dummy_y = np.ones(16)
    dummy_params = np.ones(13)
    _ = full_model_ode(0.0, dummy_y, dummy_params)
    print("Pre-compilation complete! Waking up the swarm...")
    # ==========================================
    
    start_time = time.time()
    
    # batch_size for writing to CSV
    batch_size = 5000 
    batch_results = []
    
    # Use multiprocessing.Pool instead of ProcessPoolExecutor
    with mp.Pool(processes=max_workers) as pool:
        
        # imap_unordered streams the data! 
        # chunksize=100 means each core grabs 100 rows at a time (highly efficient)
        results_iterator = pool.imap_unordered(
            process_single_row, 
            data_matrix, 
            chunksize=100 
        )
        
        # Wrap the iterator in tqdm for the progress bar
        with tqdm(total=total_rows, desc="Solving ODEs", unit="row") as pbar:
            for result in results_iterator:
                batch_results.append(result)
                pbar.update(1)
                
                # Flush to CSV periodically so you don't run out of memory
                if len(batch_results) >= batch_size:
                    pd.DataFrame(batch_results, columns=headers).to_csv(output_file, mode='a', header=False, index=False)
                    batch_results = []

    # Final flush for any remaining rows
    if batch_results:
        pd.DataFrame(batch_results,columns=headers).to_csv(output_file, mode='a', header=False, index=False)
            
    end_time = time.time()
    print(f"\nSimulations complete in {end_time - start_time:.2f} seconds.")