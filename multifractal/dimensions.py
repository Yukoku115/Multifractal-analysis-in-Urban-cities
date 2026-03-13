# multifractal/dimensions.py
"""
Regression, quality filters, and dimension extraction.

    extract_dimensions(matrices, r2_pos, r2_neg) → dims dict
"""

import numpy as np
from scipy.stats import linregress


def extract_dimensions(matrices: dict,
                       r2_thresh_pos: float = 0.97,
                       r2_thresh_neg: float = 0.89) -> dict:
    """
    Run OLS regression on box-counting matrices, apply quality filters,
    and extract D0, D1, D2, delta_alpha.

    Filters applied in order:
      1. R² threshold  (split positive / negative q)
      2. Space constraint  D_q <= 2.0,  f(alpha) <= 2.0
      3. Monotonicity  alpha must decrease as q increases

    Parameters
    ----------
    matrices      : output of run_boxcount()
    r2_thresh_pos : R² threshold for q >= 0
    r2_thresh_neg : R² threshold for q < 0

    Returns
    -------
    dims : dict with keys
        q_plot, alpha_plot, f_plot, tau_plot   (filtered arrays)
        D0, D1, D2, delta_alpha
        peak_idx
        violations  (monotonicity violations removed)
        n_valid, n_total
    """
    tau_q_matrix = matrices['tau_q_matrix']
    num_alpha    = matrices['num_alpha']
    num_f        = matrices['num_f']
    log_eps      = matrices['log_eps']
    q_values     = matrices['q_values']
    n_q          = len(q_values)

    valid_q, valid_alpha, valid_f, valid_tau, valid_r2 = [], [], [], [], []

    for j, q in enumerate(q_values):
        vm = ~np.isnan(tau_q_matrix[j])
        if vm.sum() < 4:
            continue

        s_a, _, r_a, _, _ = linregress(log_eps[vm], num_alpha[j, vm])
        s_f, _, r_f, _, _ = linregress(log_eps[vm], num_f[j, vm])
        s_t, _, r_t, _, _ = linregress(log_eps[vm], tau_q_matrix[j, vm])

        # Gate 1 — R²
        r2_min    = min(r_a**2, r_f**2, r_t**2)
        threshold = r2_thresh_neg if q < 0 else r2_thresh_pos
        if r2_min < threshold:
            continue

        # Gate 2 — space constraint
        if not np.isclose(q, 1.0):
            if s_t / (q - 1) > 2.0:
                continue
        if s_f > 2.0:
            continue

        valid_q.append(q); valid_alpha.append(s_a)
        valid_f.append(s_f); valid_tau.append(s_t); valid_r2.append(r2_min)

    # Gate 3 — monotonicity
    combined = sorted(zip(valid_q, valid_alpha, valid_f, valid_tau, valid_r2))
    q_mono, a_mono, f_mono, t_mono = [], [], [], []
    prev_alpha = np.inf
    violations = 0
    for q_, a_, f_, t_, _ in combined:
        if a_ < prev_alpha:
            q_mono.append(q_); a_mono.append(a_)
            f_mono.append(f_); t_mono.append(t_)
            prev_alpha = a_
        else:
            violations += 1

    q_plot     = np.array(q_mono)
    alpha_plot = np.array(a_mono)
    f_plot     = np.array(f_mono)
    tau_plot   = np.array(t_mono)

    n_valid = len(q_plot)
    print(f"[dimensions] {n_valid}/{n_q} q-values passed all filters "
          f"({violations} monotonicity violations removed)")

    # ── Dimension extraction ──────────────────────────────────────────────────
    if n_valid > 0:
        peak_idx    = np.argmax(f_plot)
        D0          = f_plot[peak_idx]
        D1          = alpha_plot[np.argmin(np.abs(q_plot - 1.0))]
        D2          = f_plot[np.argmin(np.abs(q_plot - 2.0))]
        delta_alpha = alpha_plot.max() - alpha_plot.min()

        if abs(D0 - D1) < 0.005:
            print("    ⚠ WARNING: D0 ≈ D1 — spectrum may be degenerate")
        if D0 < D1 or D1 < D2:
            print("    ⚠ WARNING: Hierarchy D0 ≥ D1 ≥ D2 violated")

        print(f"    D0={D0:.4f}  D1={D1:.4f}  D2={D2:.4f}  Δα={delta_alpha:.4f}")
    else:
        D0 = D1 = D2 = delta_alpha = 0.0
        peak_idx = 0
        print("    ERROR: No valid q-values found.")

    return dict(
        q_plot=q_plot, alpha_plot=alpha_plot,
        f_plot=f_plot, tau_plot=tau_plot,
        D0=D0, D1=D1, D2=D2,
        delta_alpha=delta_alpha,
        peak_idx=peak_idx,
        violations=violations,
        n_valid=n_valid, n_total=n_q,
        r2_thresh_pos=r2_thresh_pos,
        r2_thresh_neg=r2_thresh_neg,
    )
