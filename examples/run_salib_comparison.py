"""SALib ground truth comparison for Friedman-1.

Computes true Sobol indices via Saltelli sampling and compares to
HiFi-ANOVA recovery at GCV-optimal hyperparameters.
"""
import jax; jax.config.update('jax_enable_x64', True)
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import jax.numpy as jnp
import sys; sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.makedirs('figures', exist_ok=True)

from SALib.sample import saltelli
from SALib.analyze import sobol as salib_sobol

from hifi_anova.data.test_functions import T2_1_friedman1
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.core.features import build_first_order_features, build_second_order_features
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.hyperopt import optimize_multi_lambda, ridge_solve_with_diagnostics
from hifi_anova.training.regularization import build_regularization_vector


# ============================================================
# Step 1: SALib ground truth
# ============================================================
print("="*70)
print("STEP 1: SALib Ground Truth (Saltelli sampling, N=2^14)")
print("="*70)

D = 10
problem = {
    'num_vars': D,
    'names': [f'x{i+1}' for i in range(D)],
    'bounds': [[0, 1]] * D,
}

def friedman1_func(X):
    return (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20.0 * (X[:, 2] - 0.5)**2
            + 10.0 * X[:, 3]
            + 5.0 * X[:, 4])

X_saltelli = saltelli.sample(problem, 2**14, calc_second_order=True)
Y_saltelli = friedman1_func(X_saltelli)
Si = salib_sobol.analyze(problem, Y_saltelli, calc_second_order=True)

print("\nSALib First-Order Sobol (S1):")
for i in range(D):
    print(f"  x{i+1}: {Si['S1'][i]:.4f} +/- {Si['S1_conf'][i]:.4f}")

print("\nSALib Total-Order Sobol (ST):")
for i in range(D):
    print(f"  x{i+1}: {Si['ST'][i]:.4f} +/- {Si['ST_conf'][i]:.4f}")

print("\nSALib Second-Order Sobol (S2, top pairs):")
s2_flat = {}
for i in range(D):
    for j in range(i+1, D):
        s2_flat[(i,j)] = Si['S2'][i,j]
top_pairs = sorted(s2_flat.items(), key=lambda x: -x[1])[:5]
for (i,j), val in top_pairs:
    print(f"  (x{i+1},x{j+1}): {val:.4f} +/- {Si['S2_conf'][i,j]:.4f}")


# ============================================================
# Step 2: HiFi-ANOVA with GCV-optimal hyperparameters
# ============================================================
print("\n" + "="*70)
print("STEP 2: HiFi-ANOVA with GCV-Optimal Hyperparameters")
print("="*70)

X_data, y_data, _ = T2_1_friedman1(n_samples=10000, noise_std=1.0, seed=42)
data = preprocess_data(X_data, y_data, seed=42)

x_train = np.asarray(data['x_train'])
y_train = np.asarray(data['y_train'])
K1 = 10
K2 = 5

phi1 = np.asarray(build_first_order_features(jnp.array(x_train), K1))
pm = PairManager(D)
phi2 = np.asarray(build_second_order_features(jnp.array(x_train), K2, pm.pair_indices))
Phi = np.concatenate([phi1, phi2], axis=1)
y_c = y_train - np.mean(y_train)

print(f"\nFeatures: N={Phi.shape[0]}, F={Phi.shape[1]}")

# GCV optimization (curvature strategy)
print("\nOptimizing lambda via GCV (curvature regularization)...")
result = optimize_multi_lambda(Phi, y_c, D, K1, K2, pm.P, 'curvature', method='gcv')
print(f"  GCV optimal: lambda1={result['lambda_order1']:.6f}, lambda2={result['lambda_order2']:.6f}")
print(f"  df={result['df']:.1f}, MSE={result['mse']:.4f}")

# Solve at optimal
w = result['w']
G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)
G2 = np.asarray(build_gram_matrix_2d(build_gram_matrix(K2)), dtype=np.float64)
block1 = 2 * K1 + 1
block2 = (2 * K2 + 1) ** 2
F1 = D * block1

# Extract Sobol indices
first_order_vars = {}
for i in range(D):
    wi = w[i*block1:(i+1)*block1]
    first_order_vars[i] = max(0.0, float(wi @ G1 @ wi))

second_order_vars = {}
for p in range(pm.P):
    wp = w[F1 + p*block2: F1 + (p+1)*block2]
    var_p = max(0.0, float(wp @ G2 @ wp))
    i, j = pm.pair_to_variables(p)
    second_order_vars[(i,j)] = var_p

total_var = sum(first_order_vars.values()) + sum(second_order_vars.values())
hifi_anova_s1 = {i: first_order_vars[i]/total_var for i in range(D)}
hifi_anova_s2 = {k: v/total_var for k, v in second_order_vars.items()}

# Total-order: first-order + all pairs involving variable i
hifi_anova_st = {}
for i in range(D):
    t = first_order_vars[i]
    for (a,b), v in second_order_vars.items():
        if a == i or b == i:
            t += v
    hifi_anova_st[i] = t / total_var


# ============================================================
# Step 3: Comparison table
# ============================================================
print("\n" + "="*70)
print("STEP 3: SALib vs HiFi-ANOVA Comparison")
print("="*70)

print(f"\n{'Var':<6} {'SALib S1':<10} {'HiFi S1':<10} {'Err':<8} {'SALib ST':<10} {'HiFi ST':<10} {'Err':<8}")
print("-" * 62)
for i in range(D):
    s1_true = Si['S1'][i]
    s1_hifi_anova = hifi_anova_s1[i]
    err1 = abs(s1_true - s1_hifi_anova)
    st_true = Si['ST'][i]
    st_hifi_anova = hifi_anova_st[i]
    errt = abs(st_true - st_hifi_anova)
    active = "*" if s1_true > 0.02 else ""
    print(f"  x{i+1}{active:<3} {s1_true:<10.4f} {s1_hifi_anova:<10.4f} {err1:<8.4f} {st_true:<10.4f} {st_hifi_anova:<10.4f} {errt:<8.4f}")

print(f"\nTop second-order interactions:")
print(f"{'Pair':<12} {'SALib S2':<10} {'HiFi S2':<10} {'Err':<8}")
print("-" * 40)
for (i,j), val in top_pairs:
    hifi_anova_val = hifi_anova_s2.get((i,j), 0.0)
    err = abs(val - hifi_anova_val)
    print(f"  (x{i+1},x{j+1}){'':<5} {val:<10.4f} {hifi_anova_val:<10.4f} {err:<8.4f}")

# Summary statistics
s1_errors = [abs(Si['S1'][i] - hifi_anova_s1[i]) for i in range(D)]
st_errors = [abs(Si['ST'][i] - hifi_anova_st[i]) for i in range(D)]
active_s1_errors = [abs(Si['S1'][i] - hifi_anova_s1[i]) for i in range(5)]
active_st_errors = [abs(Si['ST'][i] - hifi_anova_st[i]) for i in range(5)]

print(f"\nSummary:")
print(f"  Mean absolute error (S1, all vars):    {np.mean(s1_errors):.4f}")
print(f"  Mean absolute error (S1, active only):  {np.mean(active_s1_errors):.4f}")
print(f"  Mean absolute error (ST, active only):  {np.mean(active_st_errors):.4f}")
print(f"  Max irrelevant S1:                      {max(hifi_anova_s1[i] for i in range(5,10)):.5f}")
print(f"  S12 (x1,x2 interaction):                SALib={Si['S2'][0,1]:.4f}, HiFi-ANOVA={hifi_anova_s2.get((0,1),0):.4f}")


# ============================================================
# Step 4: Comparison bar plot
# ============================================================
print("\n" + "="*70)
print("STEP 4: Generating comparison figure")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# First-order comparison
ax = axes[0]
x_pos = np.arange(D)
width = 0.35
bars1 = ax.bar(x_pos - width/2, Si['S1'], width, label='SALib (ground truth)',
               color='steelblue', alpha=0.8, yerr=Si['S1_conf'], capsize=3)
bars2 = ax.bar(x_pos + width/2, [hifi_anova_s1[i] for i in range(D)], width,
               label='HiFi-ANOVA (GCV-optimal)', color='coral', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'x{i+1}' for i in range(D)])
ax.set_ylabel('First-Order Sobol Index')
ax.set_title('First-Order Sobol: SALib vs HiFi-ANOVA')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Total-order comparison
ax = axes[1]
bars1 = ax.bar(x_pos - width/2, Si['ST'], width, label='SALib (ground truth)',
               color='steelblue', alpha=0.8, yerr=Si['ST_conf'], capsize=3)
bars2 = ax.bar(x_pos + width/2, [hifi_anova_st[i] for i in range(D)], width,
               label='HiFi-ANOVA (GCV-optimal)', color='coral', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'x{i+1}' for i in range(D)])
ax.set_ylabel('Total-Order Sobol Index')
ax.set_title('Total-Order Sobol: SALib vs HiFi-ANOVA')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/salib_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: figures/salib_comparison.png")


# ============================================================
# Step 5: Regularization path with GCV optimal marked
# ============================================================
print("\n" + "="*70)
print("STEP 5: Regularization path with GCV optimal")
print("="*70)

from hifi_anova.analysis.reg_path import compute_reg_path, plot_reg_path

path = compute_reg_path(
    Phi, y_c, D, K1, K2, pm.P,
    pair_indices=np.asarray(pm.pair_indices),
    strategy='curvature', lambda_ratio=10.0,
    n_lambdas=50, lambda_range=(1e-5, 10.0),
)

fig = plot_reg_path(path, save_prefix='figures/friedman1_gcv')
print(f"GCV optimal lambda: {path.lambda_gcv_opt:.6f}")
print("Saved: figures/friedman1_gcv_reg_path.png")

print("\nDONE.")
