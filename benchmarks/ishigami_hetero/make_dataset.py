"""Regenerate the fixed heteroscedastic-Ishigami benchmark CSVs (deterministic).

Run from the repo root:  python benchmarks/ishigami_hetero/make_dataset.py

Produces (all committed to the repo, so the benchmark is fixed):
  train.csv       x1,x2,x3,y                     — fit your model on this
  test.csv        x1,x2,x3,y                     — evaluate predictions here
  test_truth.csv  x1,x2,x3,f_true,sigma_true     — noiseless mean + true noise std
                                                    (for diagnosing mean recovery)

The inputs are x_i ~ U(-pi, pi); the response is the Ishigami function
    f(x) = sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1)
with heteroscedastic Gaussian noise whose std ramps 0.3 -> 3.0 across x3. x3 has
NO first-order effect on the mean (it acts only through the x1-x3 interaction),
yet it is the sole driver of the noise variance — a hidden heteroscedastic driver.
"""

import os
import numpy as np

from hifi_anova.data.synthetic import generate_ishigami, ishigami_sobol_indices

HERE = os.path.dirname(os.path.abspath(__file__))
A, B = 7.0, 0.1
SIGMA_MIN, SIGMA_MAX = 0.3, 3.0
N_TRAIN, N_TEST = 2000, 5000


def _f_true(X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + A * np.sin(x2) ** 2 + B * (x3 ** 4) * np.sin(x1)


def _save(path, arr, header):
    np.savetxt(path, arr, delimiter=',', header=header, comments='',
               fmt='%.10g')
    print(f"  wrote {os.path.relpath(path)}  ({arr.shape[0]} rows)")


def main():
    Xtr, ytr, _ = generate_ishigami(
        n_samples=N_TRAIN, a=A, b=B, heteroscedastic=True, variance_variable=2,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=20240517)
    Xte, yte, sig_te = generate_ishigami(
        n_samples=N_TEST, a=A, b=B, heteroscedastic=True, variance_variable=2,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=987654321)
    f_te = _f_true(Xte)

    _save(os.path.join(HERE, 'train.csv'),
          np.column_stack([Xtr, ytr]), 'x1,x2,x3,y')
    _save(os.path.join(HERE, 'test.csv'),
          np.column_stack([Xte, yte]), 'x1,x2,x3,y')
    _save(os.path.join(HERE, 'test_truth.csv'),
          np.column_stack([Xte, f_te, sig_te]), 'x1,x2,x3,f_true,sigma_true')

    gt = ishigami_sobol_indices(A, B)
    print("\nGround-truth first-order Sobol (the sensitivity target):")
    print(f"  x1={gt['first_order'][0]:.4f}  x2={gt['first_order'][1]:.4f}  "
          f"x3={gt['first_order'][2]:.4f}")
    print(f"  total-order x3 = {gt['total_order'][2]:.4f} (pure x1-x3 interaction)")


if __name__ == '__main__':
    main()
