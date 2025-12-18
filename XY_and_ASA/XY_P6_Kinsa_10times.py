# ==========================================================
# Standard XY-QAOA: Uniform Mixer (Mixed P=6) - Batch Experiment
# Setup: P=6 (6 Players), Split into 2 Teams of 3 (G=2, N=3)
# Logic: Uniform Mixing (Standard Approach)
# With: Automated Solution Quality Analysis & CSV Export (Batch)
# ==========================================================

import os
import time
import csv
import itertools
import numpy as np
from datetime import datetime
from collections import defaultdict
from qiskit import QuantumCircuit
from qiskit_braket_provider import BraketProvider
from qiskit_algorithms.optimizers import COBYLA

# -------------------------
# ユーザー設定
# -------------------------
PROBLEM_CONFIG = {'P': 6, 'G': 2, 'R': 2, 'N': 3}

QPU_BACKEND_NAME = "SV1"
P_STEPS = 1        # ※本番比較時は 5 推奨
MAXITER = 50       # ※本番比較時は 20~50 推奨
SHOTS_EVAL = 1024
SHOTS_FINAL = 4096
# 制約項の重み
QUBO_LAMBDAS = (0.1, 1.0, 0.1, 1.0, 1.0) 
DEVICE_REGION = {"SV1": "us-west-1"}

# ★何回実験するか
NUM_EXPERIMENTS = 10
BASE_SEED = 314  # ここから +1 していく

# ==========================================================
# 共通ユーティリティ
# ==========================================================
def var_indexer(P, G, R):
    def idx(p, g, r): return p * (G * R) + g * R + r
    return idx, P * G * R

def build_qubo(P, G, R, N, x, player_expertise, lambdas):
    idx, n_vars = var_indexer(P, G, R)
    h, J = defaultdict(float), defaultdict(float)
    lam1, lam2, lam3, lam4, lam5 = lambdas
    max_strength = np.max(x) if np.max(x) > 0 else 1.0
    x_norm = x / max_strength

    for g in range(G):
        for p, r in itertools.product(range(P), range(R)):
            i = idx(p, g, r); h[i] += lam1 * (x_norm[p, r] ** 2)
        for p1, r1 in itertools.product(range(P), range(R)):
            for p2, r2 in itertools.product(range(P), range(R)):
                if p1 * R + r1 >= p2 * R + r2: continue
                i, j = idx(p1, g, r1), idx(p2, g, r2)
                J[(i, j)] += lam1 * 2.0 * x_norm[p1, r1] * x_norm[p2, r2]
    for g, r in itertools.product(range(G), range(R)):
        for p1, p2 in itertools.combinations(range(P), 2):
            J[(idx(p1, g, r), idx(p2, g, r))] += lam2
    for p, g, r in itertools.product(range(P), range(G), range(R)):
        h[idx(p, g, r)] -= lam3 * player_expertise[p, r]
    for p in range(P):
        vars_p = [idx(p, g, r) for g, r in itertools.product(range(G), range(R))]
        for i in vars_p: h[i] -= lam4
        for i, j in itertools.combinations(vars_p, 2): J[(i, j)] += 2 * lam4
    for g in range(G):
        vars_g = [idx(p, g, r) for p, r in itertools.product(range(P), range(R))]
        for i in vars_g: h[i] += lam5 * (1 - 2 * N)
        for i, j in itertools.combinations(vars_g, 2): J[(i, j)] += 2 * lam5
    return h, J

def qubo_to_ising_terms(h, J, nvar):
    c0 = 0.0; z_lin = np.zeros(nvar); pairs = defaultdict(float)
    for i, v in h.items():
        c0 += 0.5 * v; z_lin[i] -= 0.5 * v
    for (i, j), v in J.items():
        c0 += 0.25 * v; z_lin[i] -= 0.25 * v; z_lin[j] -= 0.25 * v; pairs[(i, j)] = 0.25 * v
    return c0, z_lin, pairs

def energy_of_bitstring(bitstring, c0, z_lin, pairs, n_qubits):
    if len(bitstring) < n_qubits: return float('inf')
    b = bitstring[:n_qubits][::-1]
    z = np.array([1 if b_i == '0' else -1 for b_i in b], dtype=int)
    energy = c0 + np.dot(z_lin, z)
    for (i, j), v in pairs.items(): energy += v * z[i] * z[j]
    return float(energy)

def check_validity(bitstring, P, G, R, N):
    idx, n_vars = var_indexer(P, G, R)
    if len(bitstring) < n_vars: return "err_short"
    b = bitstring[:n_vars][::-1]
    for p in range(P):
        if sum(b[idx(p, g, r)] == '1' for g, r in itertools.product(range(G), range(R))) != 1: return f"fail_P{p}"
    for g in range(G):
        if sum(b[idx(p, g, r)] == '1' for p, r in itertools.product(range(P), range(R))) != N: return f"fail_G{g}"
    return "valid"

def decode_solution(bitstring, P, G, R, strength_ratings, expertise_data):
    strength_flat = strength_ratings.flatten(); idx, n_vars = var_indexer(P, G, R)
    b = bitstring[:n_vars][::-1]; assignments = defaultdict(list)
    for p, g, r in itertools.product(range(P), range(G), range(R)):
        if b[idx(p, g, r)] == '1':
            s = strength_flat[p]; e = expertise_data[p, r]
            assignments[f"Group {g}"].append(f"P{p} (Rating:{s:.1f}, Exp:{e}, Str:{s*e:.1f})")
    return dict(sorted(assignments.items()))

# ==========================================================
# Standard XY-QAOA Logic
# ==========================================================

def create_valid_initial_state(n_qubits, P, G, R):
    qc = QuantumCircuit(n_qubits)
    idx_func, _ = var_indexer(P, G, R)
    current_p = 0
    for g in range(G):
        for _ in range(PROBLEM_CONFIG['N']): 
            if current_p < P:
                r = current_p % R 
                target = idx_func(current_p, g, r)
                qc.x(target)
                current_p += 1
    return qc

def calculate_uniform_weights(P):
    """ Standard XY-QAOA: Uniform Weights (1.0) """
    weights = {}
    for p1, p2 in itertools.combinations(range(P), 2):
        weights[(p1, p2)] = 1.0
        weights[(p2, p1)] = 1.0
    return weights

def append_xy_mixer(qc, beta, n_qubits, P, G, R, mixer_weights):
    idx_func, _ = var_indexer(P, G, R)
    for p1, p2 in itertools.combinations(range(P), 2):
        w = mixer_weights.get((p1, p2), 1.0)
        for g in range(G):
            for r in range(R):
                q1 = idx_func(p1, g, r)
                q2 = idx_func(p2, g, r)
                qc.rxx(2 * beta * w, q1, q2)
                qc.ryy(2 * beta * w, q1, q2)

# ==========================================================
# Backend Helper
# ==========================================================
def get_backend(provider, requested_name: str):
    region = DEVICE_REGION.get(requested_name)
    if region: os.environ["AWS_DEFAULT_REGION"] = region
    try: return provider.get_backend(requested_name)
    except Exception as e: raise RuntimeError(f"Error: {e}")

# ==========================================================
# 1回の実験を行う関数 (seedを受け取る)
# ==========================================================
def run_single_experiment(seed_val, Pn, Gn, Rn, N, provider, strength_ratings, player_expertise, h, J, c0, z_lin, pairs, n_qubits):
    np.random.seed(seed_val) # ★ここでSEEDを設定
    
    try: backend = get_backend(provider, QPU_BACKEND_NAME)
    except: return None
    
    initial_qc = create_valid_initial_state(n_qubits, Pn, Gn, Rn)
    mixer_weights = calculate_uniform_weights(Pn)

    def objective_function(params):
        p_s = len(params) // 2
        beta, gamma = params[:p_s], params[p_s:]
        qc = QuantumCircuit(n_qubits)
        qc.compose(initial_qc, inplace=True)
        for i in range(p_s):
            for j, zi in enumerate(z_lin):
                if abs(zi)>1e-5: qc.rz(2*gamma[i]*zi, j)
            for (j, k), zz in pairs.items():
                if abs(zz)>1e-5: qc.rzz(2*gamma[i]*zz, j, k)
            append_xy_mixer(qc, beta[i], n_qubits, Pn, Gn, Rn, mixer_weights)
        qc.measure_all()
        try:
            counts = backend.run(qc, shots=SHOTS_EVAL).result().get_counts()
        except: return 1e5
        avg_e = 0
        total_shots = sum(counts.values())
        if total_shots == 0: return 1e5
        for b, c in counts.items():
            avg_e += energy_of_bitstring(b, c0, z_lin, pairs, n_qubits) * c
        return avg_e / total_shots

    init_params = np.concatenate([np.ones(P_STEPS)*0.05, np.zeros(P_STEPS)])
    
    # Optimize
    start_time = time.time()
    opt = COBYLA(maxiter=MAXITER, tol=0.001)
    res = opt.minimize(objective_function, x0=init_params)
    elapsed = time.time() - start_time
    
    # Final Eval
    qc_final = QuantumCircuit(n_qubits)
    qc_final.compose(initial_qc, inplace=True)
    beta_opt, gamma_opt = res.x[:P_STEPS], res.x[P_STEPS:]
    for i in range(P_STEPS):
        for j, zi in enumerate(z_lin):
            if abs(zi)>1e-5: qc_final.rz(2*gamma_opt[i]*zi, j)
        for (j, k), zz in pairs.items():
            if abs(zz)>1e-5: qc_final.rzz(2*gamma_opt[i]*zz, j, k)
        append_xy_mixer(qc_final, beta_opt[i], n_qubits, Pn, Gn, Rn, mixer_weights)
    qc_final.measure_all()
    final_counts = backend.run(qc_final, shots=SHOTS_FINAL).result().get_counts()
    
    best_sol = ""; best_e = float('inf'); valid_count = 0
    for b, c in final_counts.items():
        if check_validity(b, Pn, Gn, Rn, N) == "valid":
            valid_count += c
            e = energy_of_bitstring(b, c0, z_lin, pairs, n_qubits)
            if e < best_e: best_e = e; best_sol = b
            
    # Metrics
    valid_ratio = valid_count / SHOTS_FINAL
    success_prob = 0.0
    
    # ---------------------------------------------------------
    # チーム分け文字列 "025/134" の生成
    # ---------------------------------------------------------
    team_str_simple = "None"
    if best_sol:
        success_count = final_counts.get(best_sol, 0)
        success_prob = success_count / SHOTS_FINAL
        
        # decode関数で解析
        team_dict = decode_solution(best_sol, Pn, Gn, Rn, strength_ratings, player_expertise)
        
        # 文字列整形処理
        groups = []
        for g_name in sorted(team_dict.keys()): 
            members = team_dict[g_name]
            p_indices = [m.split()[0].replace('P','') for m in members]
            p_indices.sort()
            groups.append("".join(p_indices))
        team_str_simple = "/".join(groups)

    return {
        "seed": seed_val,
        "valid_ratio": valid_ratio,
        "success_prob": success_prob,
        "best_energy": best_e,
        "time": elapsed,
        "iterations": res.nfev,
        "best_team": team_str_simple
    }

# ==========================================================
# Main Batch Execution
# ==========================================================
def main():
    print(f"\n▶ Starting Batch Experiment: Standard XY-QAOA")
    print(f"   Runs: {NUM_EXPERIMENTS} times (Varying SEED)")
    print(f"   Params: Steps={P_STEPS}, MaxIter={MAXITER}")
    
    Pn, Gn, Rn, N = PROBLEM_CONFIG.values()
    provider = BraketProvider()
    n_qubits = Pn * Gn * Rn
    strength_ratings = np.array([[2.0], [1.2], [1.8], [1.4], [2.2], [0.9]])
    player_expertise = np.array([[5, 1], [1, 5], [5, 1], [1, 5], [4, 2], [1, 1]])
    x = strength_ratings * player_expertise
    h, J = build_qubo(Pn, Gn, Rn, N, x, player_expertise, QUBO_LAMBDAS)
    c0, z_lin, pairs = qubo_to_ising_terms(h, J, n_qubits)
    c0 += (QUBO_LAMBDAS[3] * Pn + QUBO_LAMBDAS[4] * Gn * (N ** 2))

    # Store results
    results = []
    
    for i in range(NUM_EXPERIMENTS):
        current_seed = BASE_SEED + i
        print(f"   Running experiment {i+1}/{NUM_EXPERIMENTS} (Seed={current_seed})...", end="", flush=True)
        try:
            res = run_single_experiment(current_seed, Pn, Gn, Rn, N, provider, strength_ratings, player_expertise, h, J, c0, z_lin, pairs, n_qubits)
            if res:
                results.append(res)
                print(f" Done. (Team={res['best_team']}, E={res['best_energy']:.3f})")
            else:
                print(" Failed (Backend error)")
        except Exception as e:
            print(f" Failed! {e}")

    # ---------------------------------------------------------
    # 統計処理 & CSV出力
    # ---------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"XY_10times_P6_{timestamp_str}.csv"
    
    # 平均値の計算
    if results:
        avg_valid = np.mean([r['valid_ratio'] for r in results])
        std_valid = np.std([r['valid_ratio'] for r in results])
        avg_success = np.mean([r['success_prob'] for r in results])
        std_success = np.std([r['success_prob'] for r in results])
        avg_energy = np.mean([r['best_energy'] for r in results])
        std_energy = np.std([r['best_energy'] for r in results])

        print(f"\n=== Batch Experiment Results (N={NUM_EXPERIMENTS}) ===")
        print(f"1. Valid Ratio:       {avg_valid*100:.2f}% ± {std_valid*100:.2f}%")
        print(f"2. Success Prob:      {avg_success*100:.2f}% ± {std_success*100:.2f}%")
        print(f"3. Best Energy:       {avg_energy:.4f} ± {std_energy:.4f}")
        
        # CSV書き込み
        header = ["Run", "Seed", "Best Team", "Valid Ratio", "Success Prob", "Best Energy", "Time(s)", "Iterations"]
        
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, r in enumerate(results):
                writer.writerow([
                    i+1, 
                    r['seed'], 
                    r['best_team'], 
                    f"{r['valid_ratio']:.4f}",
                    f"{r['success_prob']:.4f}", 
                    f"{r['best_energy']:.4f}", 
                    f"{r['time']:.2f}", 
                    r['iterations']
                ])
            # 平均行
            writer.writerow([])
            writer.writerow(["AVERAGE", "-", "-", f"{avg_valid:.4f}", f"{avg_success:.4f}", f"{avg_energy:.4f}", "-", "-"])
            writer.writerow(["STD_DEV", "-", "-", f"{std_valid:.4f}", f"{std_success:.4f}", f"{std_energy:.4f}", "-", "-"])
            
        print(f"\n[Saved]: {filename}")
    else:
        print("\n[Error]: No results to save.")

if __name__ == "__main__":
    main()