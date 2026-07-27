"""Compare prediction-optimal vs Sobol-estimation vs SALib ground truth.

Demonstrates the two-mode approach:
- Mode 1 (Prediction): GCV-optimal lambda, good MSE, shrunk Sobol
- Mode 2 (Sobol estimation): additivity-optimal lambda, unbiased Sobol
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
from hifi_anova.training.hyperopt import optimize_multi_lambda
from hifi_anova.training.trainer import estimate_sobol
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.ridge import weighted_ridge_solve


# ============================================================
# SALib ground truth
# ============================================================
D = 10
problem = {'num_vars': D, 'names': [f'x{i+1}' for i in range(D)],
           'bounds': [[0,1]]*D}
X_sal = saltelli.sample(problem, 2**14, calc_second_order=True)
Y_sal = (10*np.sin(np.pi*X_sal[:,0]*X_sal[:,1]) + 20*(X_sal[:,2]-0.5)**2
         + 10*X_sal[:,3] + 5*X_sal[:,4])
Si = salib_sobol.analyze(problem, Y_sal, calc_second_order=True)


# ============================================================
# Generate data
# ============================================================
X, y, _ = T2_1_friedman1(n_samples=10000, noise_std=1.0, seed=42)
data = preprocess_data(X, y, seed=42)
x_train = np.asarray(data['x_train'])
y_train = np.asarray(data['y_train'])
K1, K2 = 10, 5
pm = PairManager(D)

phi1 = np.asarray(build_first_order_features(jnp.array(x_train), K1))
phi2 = np.asarray(build_second_order_features(jnp.array(x_train), K2, pm.pair_indices))
Phi = np.concatenate([phi1, phi2], axis=1)
y_c = y_train - np.mean(y_train)
f0 = np.mean(y_train)

G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)
G2 = np.asarray(build_gram_matrix_2d(build_gram_matrix(K2)), dtype=np.float64)
block1, block2 = 2*K1+1, (2*K2+1)**2
F1 = D * block1


def sobol_from_w(w):
    w = np.asarray(w, dtype=np.float64)
    fo = {}
    for i in range(D):
        wi = w[i*block1:(i+1)*block1]
        fo[i] = max(0, float(wi @ G1 @ wi))
    so = {}
    for p in range(pm.P):
        wp = w[F1+p*block2:F1+(p+1)*block2]
        v = max(0, float(wp @ G2 @ wp))
        i, j = pm.pair_to_variables(p)
        so[(i,j)] = v
    total = sum(fo.values()) + sum(so.values())
    s1 = {i: fo[i]/total for i in range(D)} if total > 0 else {}
    s2 = {k: v/total for k,v in so.items()} if total > 0 else {}
    st = {}
    for i in range(D):
        t = fo.get(i,0)
        for (a,b),v in so.items():
            if a==i or b==i: t += v
        st[i] = t/total if total > 0 else 0
    return s1, s2, st


# ============================================================
# Mode 1: Prediction-optimal (GCV)
# ============================================================
print("="*70)
print("MODE 1: Prediction-Optimal (GCV)")
print("="*70)
result_pred = optimize_multi_lambda(Phi, y_c, D, K1, K2, pm.P, 'curvature', 'gcv')
w_pred = result_pred['w']
s1_pred, s2_pred, st_pred = sobol_from_w(w_pred)
pred_rmse = np.sqrt(result_pred['mse'])

print(f"  lambda1={result_pred['lambda_order1']:.4e}, lambda2={result_pred['lambda_order2']:.4e}")
print(f"  df={result_pred['df']:.1f}, RMSE={pred_rmse:.4f}")
add_pred = sum(s1_pred.values()) + sum(s2_pred.values())
print(f"  Sobol sum (additivity): {add_pred:.4f}")


# ============================================================
# Mode 2: Sobol estimation (additivity-optimal)
# ============================================================
print("\n" + "="*70)
print("MODE 2: Sobol Estimation (Additivity-Optimal)")
print("="*70)
sobol_result = estimate_sobol(
    jnp.array(x_train), jnp.array(y_train),
    K1=K1, K2=K2, strategy='curvature', auto_lambda=True)
s1_sobol = sobol_result['sobol_first_order']
s2_sobol = sobol_result['sobol_second_order']
st_sobol = sobol_result['sobol_total_order']
print(f"  lambda1={sobol_result['lambda_order1']:.4e}, lambda2={sobol_result['lambda_order2']:.4e}")
print(f"  Sobol sum (additivity): {sobol_result['additivity_sum']:.4f}")


# ============================================================
# Comparison table
# ============================================================
print("\n" + "="*70)
print("COMPARISON: SALib vs Prediction-Model Sobol vs Estimation-Mode Sobol")
print("="*70)

print(f"\n{'Var':<6} {'SALib S1':<10} {'Pred S1':<10} {'Est S1':<10} │ {'SALib ST':<10} {'Pred ST':<10} {'Est ST':<10}")
print("-"*72)
for i in range(D):
    active = "*" if Si['S1'][i] > 0.02 else " "
    print(f"  x{i+1}{active} {Si['S1'][i]:<10.4f} {s1_pred[i]:<10.4f} {s1_sobol[i]:<10.4f} "
          f"│ {Si['ST'][i]:<10.4f} {st_pred[i]:<10.4f} {st_sobol[i]:<10.4f}")

print(f"\n  Second-order interaction (x1,x2):")
print(f"    SALib:      {Si['S2'][0,1]:.4f}")
print(f"    Prediction: {s2_pred.get((0,1),0):.4f}  (shrunk by ridge)")
print(f"    Estimation: {s2_sobol.get((0,1),0):.4f}  (less shrunk)")

# Error summary
print(f"\n  {'Metric':<35} {'Prediction':<12} {'Estimation':<12}")
print(f"  {'-'*59}")
err_s1_pred = np.mean([abs(Si['S1'][i]-s1_pred[i]) for i in range(5)])
err_s1_sobol = np.mean([abs(Si['S1'][i]-s1_sobol[i]) for i in range(5)])
err_st_pred = np.mean([abs(Si['ST'][i]-st_pred[i]) for i in range(5)])
err_st_sobol = np.mean([abs(Si['ST'][i]-st_sobol[i]) for i in range(5)])
err_s12_pred = abs(Si['S2'][0,1] - s2_pred.get((0,1),0))
err_s12_sobol = abs(Si['S2'][0,1] - s2_sobol.get((0,1),0))
print(f"  {'MAE S1 (active vars)':<35} {err_s1_pred:<12.4f} {err_s1_sobol:<12.4f}")
print(f"  {'MAE ST (active vars)':<35} {err_st_pred:<12.4f} {err_st_sobol:<12.4f}")
print(f"  {'|S12 error|':<35} {err_s12_pred:<12.4f} {err_s12_sobol:<12.4f}")
print(f"  {'Sobol sum (target=1.0)':<35} {add_pred:<12.4f} {sobol_result['additivity_sum']:<12.4f}")
print(f"  {'Prediction RMSE':<35} {pred_rmse:<12.4f} {'(N/A)':<12}")


# ============================================================
# Figure: 3-way comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

x_pos = np.arange(D)
width = 0.25

# First-order
ax = axes[0]
ax.bar(x_pos - width, Si['S1'], width, label='SALib (truth)',
       color='steelblue', alpha=0.8, yerr=Si['S1_conf'], capsize=2)
ax.bar(x_pos, [s1_pred[i] for i in range(D)], width,
       label='Prediction model', color='coral', alpha=0.8)
ax.bar(x_pos + width, [s1_sobol[i] for i in range(D)], width,
       label='Sobol estimation', color='seagreen', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'x{i+1}' for i in range(D)], fontsize=9)
ax.set_ylabel('First-Order Sobol Index')
ax.set_title('First-Order Sobol Indices')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Total-order
ax = axes[1]
ax.bar(x_pos - width, Si['ST'], width, label='SALib (truth)',
       color='steelblue', alpha=0.8, yerr=Si['ST_conf'], capsize=2)
ax.bar(x_pos, [st_pred[i] for i in range(D)], width,
       label='Prediction model', color='coral', alpha=0.8)
ax.bar(x_pos + width, [st_sobol[i] for i in range(D)], width,
       label='Sobol estimation', color='seagreen', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'x{i+1}' for i in range(D)], fontsize=9)
ax.set_ylabel('Total-Order Sobol Index')
ax.set_title('Total-Order Sobol Indices')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# S12 comparison (bar chart)
ax = axes[2]
labels = ['SALib\n(truth)', 'Prediction\nmodel', 'Sobol\nestimation']
vals = [Si['S2'][0,1], s2_pred.get((0,1),0), s2_sobol.get((0,1),0)]
colors = ['steelblue', 'coral', 'seagreen']
bars = ax.bar(labels, vals, color=colors, alpha=0.8, width=0.6)
ax.set_ylabel('S(x1,x2) Interaction Index')
ax.set_title('Key Interaction: sin(pi*x1*x2)')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/sobol_prediction_vs_estimation.png', dpi=150, bbox_inches='tight')
print("\nSaved: figures/sobol_prediction_vs_estimation.png")
