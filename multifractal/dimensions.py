# multifractal/dimensions.py
"""
Regression, quality filters, and dimension extraction.

    extract_dimensions(matrices, r2_pos, r2_neg) → dims dict

The extraction loop is a direct copy of city_multifractal_v5.py — no
refactoring, no extra gates, no monotonicity filter.
"""

import numpy as np
from scipy.stats import linregress


def extract_dimensions(matrices: dict,
                       r2_thresh_pos: float = 0.97,
                       r2_thresh_neg: float = 0.89,
                       enforce_alpha_mono: bool = False) -> dict:

    tau_q_matrix = matrices['tau_q_matrix']
    num_alpha    = matrices['num_alpha']
    num_f        = matrices['num_f']
    log_eps      = matrices['log_eps']
    q_values     = matrices['q_values']
    n_q          = len(q_values)

    # ── Exact v5 extraction loop ──────────────────────────────────────────────
    valid_q, valid_alpha, valid_f, valid_tau, valid_r2 = [], [], [], [], []

    for j, q in enumerate(q_values):
        vm = ~np.isnan(tau_q_matrix[j])
        if vm.sum() < 4:
            continue

        s_a, _, r_a, _, _ = linregress(log_eps[vm], num_alpha[j, vm])
        s_f, _, r_f, _, _ = linregress(log_eps[vm], num_f[j, vm])
        s_t, _, r_t, _, _ = linregress(log_eps[vm], tau_q_matrix[j, vm])

        current_threshold = r2_thresh_pos if q >= 0 else r2_thresh_neg

        # Ignore R² of tau(q) at q=1 (perfectly flat line → R²=0)
        if np.isclose(q, 1.0):
            r2_min = min(r_a**2, r_f**2)
        else:
            r2_min = min(r_a**2, r_f**2, r_t**2)

        if r2_min < current_threshold:
            continue

        if not np.isclose(q, 1.0):
            D_q = s_t / (q - 1)
            if q >= 0 and D_q > 2.0:
                continue

        if s_f > 2.0:
            continue

        valid_q.append(q)
        valid_alpha.append(s_a)
        valid_f.append(s_f)
        valid_tau.append(s_t)
        valid_r2.append(r2_min)

    # ── α-monotonicity filter (optional) ────────────────────────────────────
    # By the Legendre transform, α(q) must be monotonically DECREASING in q.
    # Enable via enforce_alpha_mono=True in main.py to remove violated points.
    _sort = np.argsort(valid_q)
    vq  = np.array(valid_q)[_sort]
    va  = np.array(valid_alpha)[_sort]
    vf  = np.array(valid_f)[_sort]
    vt  = np.array(valid_tau)[_sort]

    if enforce_alpha_mono:
        mono_keep = np.ones(len(vq), dtype=bool)
        for k in range(1, len(vq)):
            if va[k] > va[k - 1]:
                mono_keep[k] = False
        n_mono_removed = int((~mono_keep).sum())
        if n_mono_removed:
            print(f"[dimensions] α-monotonicity: removed {n_mono_removed} violation(s)")
    else:
        mono_keep      = np.ones(len(vq), dtype=bool)
        n_mono_removed = 0

    q_plot     = vq[mono_keep]
    alpha_plot = va[mono_keep]
    f_plot     = vf[mono_keep]
    tau_plot   = vt[mono_keep]

    n_valid = len(q_plot)
    print(f"[dimensions] q-values passing filter: {n_valid}/{n_q}")

    # ── Build D(q) = τ(q)/(q-1) consistently from tau_plot ──────────────────
    # At q=1 τ/(q-1) is 0/0; use a central finite-difference of τ instead
    # (equivalent to L'Hôpital: D(1) = dτ/dq|_{q=1}).
    # We build this array here so D0/D1/D2 all come from the SAME source,
    # avoiding the inconsistency between tau_plot and alpha_plot regressions.
    if n_valid > 0:
        peak_idx = np.argmax(f_plot)

        Dq_arr = np.full(len(q_plot), np.nan)
        for k, q in enumerate(q_plot):
            if np.isclose(q, 1.0):
                # central finite difference of tau around q=1
                if 0 < k < len(q_plot) - 1:
                    Dq_arr[k] = (tau_plot[k + 1] - tau_plot[k - 1]) / \
                                (q_plot[k + 1] - q_plot[k - 1])
                elif k == 0:
                    Dq_arr[k] = (tau_plot[1] - tau_plot[0]) / (q_plot[1] - q_plot[0])
                else:
                    Dq_arr[k] = (tau_plot[-1] - tau_plot[-2]) / (q_plot[-1] - q_plot[-2])
            else:
                Dq_arr[k] = tau_plot[k] / (q - 1)

        # Read D0, D1, D2 off the same Dq_arr via interpolation
        D0 = float(np.interp(0.0, q_plot, Dq_arr))
        D1 = float(np.interp(1.0, q_plot, Dq_arr))
        D2 = float(np.interp(2.0, q_plot, Dq_arr))

        delta_alpha = alpha_plot.max() - alpha_plot.min()
        print(f"    D0={D0:.4f}  D1={D1:.4f}  D2={D2:.4f}  Δα={delta_alpha:.4f}")
    else:
        Dq_arr   = np.array([])
        D0 = D1 = D2 = delta_alpha = 0.0
        peak_idx = 0
        print("    ERROR: No valid q-values found.")

    return dict(
        q_plot=q_plot, alpha_plot=alpha_plot,
        f_plot=f_plot, tau_plot=tau_plot,
        Dq_arr=Dq_arr,
        D0=D0, D1=D1, D2=D2,
        delta_alpha=delta_alpha,
        peak_idx=peak_idx,
        violations=n_mono_removed,
        n_valid=n_valid, n_total=n_q,
        r2_thresh_pos=r2_thresh_pos,
        r2_thresh_neg=r2_thresh_neg,
    )