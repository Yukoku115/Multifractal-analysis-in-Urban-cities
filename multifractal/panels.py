# multifractal/panels.py
"""
All plotting functions. Each takes (dims, meta, boundary, out_dir) and
saves one PNG. Call whichever panels you want from main.py.

    diagnostic(matrices, meta, out_dir)
    density(x_norm, y_norm, boundary, meta, out_dir)
    dimension_bars(dims, meta, out_dir)
    spectrum(dims, meta, out_dir)
    tau(dims, meta, out_dir)
    fq(dims, meta, out_dir)
    Dq(dims, meta, out_dir)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress

# ── Dark/neon palette ─────────────────────────────────────────────────────────
BG          = '#0a0a0f'
PANEL_BG    = '#0f0f1a'
GRID_COLOR  = '#1e1e3a'
TEXT        = '#e8e8f0'
ACCENT1     = '#ff6b35'
ACCENT2     = '#00d4ff'
ACCENT3     = '#a855f7'
NEON_YELLOW = '#f0e040'
TAU_COLOR   = '#00ffcc'
MONO_COLOR  = '#ff4466'

_DARK_RCPARAMS = {
    'text.color':      TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color':     TEXT,
    'ytick.color':     TEXT,
    'axes.edgecolor':  '#2a2a4a',
}


def _save(fig, path, facecolor=BG):
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.2,
                facecolor=facecolor)
    plt.close(fig)
    print(f"    → {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic (white background — for checking scaling quality)
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic(matrices: dict, meta: dict, out_dir: str):
    """Three-panel scaling diagnostic: log(Z_q) lines, R² heatmap, box-count slope."""
    tau_q_matrix = matrices['tau_q_matrix']
    log_eps      = matrices['log_eps']
    q_values     = matrices['q_values']
    grid_sizes   = matrices['grid_sizes']
    n_boxes      = matrices['n_boxes']
    n_q, n_g     = tau_q_matrix.shape
    Q_MIN, Q_MAX = q_values.min(), q_values.max()

    q_diag      = [v for v in [-6, -4, -2, -1, 0, 1, 2, 4, 6]
                   if Q_MIN <= v <= Q_MAX]
    colors_diag = plt.cm.RdBu_r(np.linspace(0, 1, len(q_diag)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

    ax = axes[0]
    for qi, q_val in enumerate(q_diag):
        j     = np.argmin(np.abs(q_values - q_val))
        valid = ~np.isnan(tau_q_matrix[j])
        if valid.sum() < 3:
            continue
        ax.plot(log_eps[valid], tau_q_matrix[j, valid], 'o-',
                color=colors_diag[qi], label=f'q={q_val:.0f}', ms=5, lw=1.5)
        s, b, _, _, _ = linregress(log_eps[valid], tau_q_matrix[j, valid])
        ax.plot(log_eps[valid], s*log_eps[valid]+b, '--',
                color=colors_diag[qi], alpha=0.5, lw=1)
    ax.set_xlabel('log(ε)'); ax.set_ylabel('log(Z_q)')
    ax.set_title('Scaling check: log(Z_q) vs log(ε)\n[should be straight lines]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, ls='--', alpha=0.4)

    ax = axes[1]
    r2_map = np.full((n_q, n_g - 2), np.nan)
    for j in range(n_q):
        for k in range(n_g - 2):
            subset = slice(k, k + 3)
            valid  = ~np.isnan(tau_q_matrix[j, subset])
            if valid.sum() < 3:
                continue
            _, _, r, _, _ = linregress(log_eps[subset][valid],
                                       tau_q_matrix[j, subset][valid])
            r2_map[j, k] = r**2
    im = ax.imshow(r2_map, aspect='auto', cmap='RdYlGn', vmin=0.9, vmax=1.0,
                   extent=[0, n_g-2, Q_MIN, Q_MAX], origin='lower')
    plt.colorbar(im, ax=ax, label='R²')
    ax.set_xlabel('Scale window index'); ax.set_ylabel('q')
    ax.set_title('R² of local scaling fits\n[green=reliable, red=unreliable]')
    ax.axhline(0, color='white', ls='--', lw=0.8)

    ax = axes[2]
    ax.loglog(grid_sizes, n_boxes, 'ko-', ms=7)
    s, b, r, _, _ = linregress(np.log(grid_sizes), np.log(n_boxes))
    ax.loglog(grid_sizes, np.exp(b)*grid_sizes**s, 'r--',
              label=f'slope={s:.3f}, R²={r**2:.3f}')
    ax.set_xlabel('Grid size (bins)'); ax.set_ylabel('Occupied boxes')
    ax.set_title(f'Box-counting dimension check\nD₀ slope ≈ {s:.3f}')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4, which='both')

    city, radius = meta['city'], meta['radius']
    plt.suptitle(f'Scaling Diagnostics — {city} ({radius/1000:.0f}km core)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, f"{meta['slug']}_scaling_diagnostic.png")
    _save(fig, path, facecolor='white')


# ─────────────────────────────────────────────────────────────────────────────
# A — Density map
# ─────────────────────────────────────────────────────────────────────────────

def density(x_norm, y_norm, boundary: dict, meta: dict, out_dir: str):
    """2D density map with boundary overlay showing the box-counting region."""
    plt.rcParams.update(_DARK_RCPARAMS)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#000')
    ax.set_facecolor('#000')
    ax.hist2d(x_norm, y_norm, bins=300, cmap='magma', norm=LogNorm())

    # Draw the active box-counting boundary
    lo, hi = boundary['grid_lo'], boundary['grid_hi']
    if boundary['kind'] == 'circular':
        theta = np.linspace(0, 2*np.pi, 300)
        r     = boundary['radius']
        ax.plot(0.5 + r*np.cos(theta), 0.5 + r*np.sin(theta),
                color=ACCENT2, lw=1.2, ls='--', alpha=0.6)
        ax.text(0.5, 0.5 + r + 0.01, 'box-counting region',
                color=ACCENT2, fontsize=8, ha='center', va='bottom', alpha=0.7)
    else:
        sq = plt.Rectangle((lo, lo), hi-lo, hi-lo,
                            lw=1.2, edgecolor=ACCENT2,
                            facecolor='none', ls='--', alpha=0.6)
        ax.add_patch(sq)
        ax.text(lo + 0.01, hi + 0.01, 'box-counting region',
                color=ACCENT2, fontsize=8, va='bottom', alpha=0.7)

    city, radius = meta['city'], meta['radius']
    ax.set_title(
        f"Street Intersection Density  ·  {radius/1000:.0f}km Urban Core  ·  {city}",
        fontsize=13, color=TEXT, pad=12)
    ax.set_aspect('equal')
    ax.tick_params(colors='#444466')
    for sp in ax.spines.values(): sp.set_edgecolor('#1a1a2e')

    path = os.path.join(out_dir, f"{meta['slug']}_A_density.png")
    _save(fig, path, facecolor='#000')
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# B — Dimension bars
# ─────────────────────────────────────────────────────────────────────────────

def dimension_bars(dims: dict, meta: dict, out_dir: str):
    plt.rcParams.update(_DARK_RCPARAMS)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')

    D0, D1, D2 = dims['D0'], dims['D1'], dims['D2']
    f_plot     = dims['f_plot']

    if len(f_plot) > 0:
        for idx, (val, col) in enumerate(zip([D0, D1, D2],
                                             [ACCENT1, ACCENT2, ACCENT3])):
            ax.bar(idx, val, color=col, alpha=0.18, width=0.6, zorder=1)
            ax.bar(idx, val, color=col, alpha=0.85, width=0.45,
                   edgecolor=col, linewidth=1.5, zorder=2)
            ax.text(idx, val + 0.03, f'{val:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=13, color=col)
        ax.set_xticks(range(3))
        ax.set_xticklabels(
            ['$D_0$\n(capacity)', '$D_1$\n(information)', '$D_2$\n(correlation)'],
            color=TEXT, fontsize=11)
        ax.set_xlim(-0.6, 3.0)
        ax.set_ylim(0, 2.2)
        ax.axhline(2.0, ls='--', color='#444466', lw=1.2, alpha=0.8)
        ax.text(2.85, 2.02, 'Euclidean\nlimit',
                fontsize=8, color='#666688', va='bottom', ha='right')
        ax.annotate(
            f'D₀ − D₂ = {D0-D2:.3f}', xy=(0.5, 0.06), xycoords='axes fraction',
            ha='center', fontsize=11, color=NEON_YELLOW, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#1a1a2e',
                      ec=NEON_YELLOW, lw=1.2, alpha=0.9))

    ax.set_title("Generalised Dimensions", fontsize=13, color=TEXT, pad=12)
    ax.tick_params(axis='y', colors='#666688')
    ax.yaxis.grid(True, color=GRID_COLOR, lw=0.8, ls='--')
    ax.set_axisbelow(True)

    path = os.path.join(out_dir, f"{meta['slug']}_B_dimensions.png")
    _save(fig, path)
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# C — Singularity spectrum  f(α) vs α
# ─────────────────────────────────────────────────────────────────────────────

def spectrum(dims: dict, meta: dict, out_dir: str):
    plt.rcParams.update(_DARK_RCPARAMS)
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')

    q_plot, alpha_plot, f_plot = dims['q_plot'], dims['alpha_plot'], dims['f_plot']
    D0, delta_alpha, peak_idx  = dims['D0'], dims['delta_alpha'], dims['peak_idx']

    if len(f_plot) > 1:
        # Sort by alpha for a smooth parabola — q-order ≠ alpha-order when
        # negative-q points scatter out of sequence.
        _si = np.argsort(alpha_plot)
        _a_s, _f_s = alpha_plot[_si], f_plot[_si]
        ax.plot(_a_s, _f_s, '-', color='#ffffff', alpha=0.12, lw=4, zorder=1)
        ax.plot(_a_s, _f_s, '-', color='#aaaacc', alpha=0.5, lw=1.2, zorder=2)
        norm_q = Normalize(vmin=q_plot.min(), vmax=q_plot.max())
        ax.scatter(alpha_plot, f_plot, c=q_plot, cmap=plt.cm.plasma, norm=norm_q,
                   s=100, edgecolors='white', linewidths=0.5, zorder=4, alpha=0.95)
        ax.axvspan(alpha_plot.min(), alpha_plot[peak_idx],
                   alpha=0.06, color=ACCENT1, zorder=0)
        ax.axvspan(alpha_plot[peak_idx], alpha_plot.max(),
                   alpha=0.06, color=ACCENT2, zorder=0)
        q0_idx = np.argmin(np.abs(q_plot))
        ax.axvline(alpha_plot[q0_idx], color='#444466', ls=':', lw=1, alpha=0.6)
        cb = fig.colorbar(ScalarMappable(norm=norm_q, cmap=plt.cm.plasma),
                          ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('q value', color=TEXT, fontsize=10)
        cb.ax.yaxis.set_tick_params(color=TEXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
        cb.outline.set_edgecolor('#2a2a4a')
        ax.annotate(f'D₀ = {D0:.3f}',
                    xy=(alpha_plot[peak_idx], D0),
                    xytext=(0.45, 0.90), textcoords='axes fraction',
                    color=NEON_YELLOW, fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=NEON_YELLOW, lw=1.2))
        ax.annotate(f'Δα = {delta_alpha:.3f}',
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    color=ACCENT2, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e',
                              ec=ACCENT2, lw=1.0, alpha=0.9))
    else:
        ax.text(0.5, 0.5, f'Insufficient data ({len(f_plot)} point(s))',
                ha='center', va='center', transform=ax.transAxes,
                color=TEXT, fontsize=12)

    ax.axhline(2.0, ls='--', color='#444466', lw=1, alpha=0.6)
    ax.set_title(r"Singularity Spectrum  $f(\alpha)$", fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel(r"Singularity strength  $\alpha$", fontsize=12, color=TEXT)
    ax.set_ylabel(r"Fractal dimension  $f(\alpha)$", fontsize=12, color=TEXT)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors='#666688')
    ax.grid(True, color=GRID_COLOR, lw=0.8, ls='--')
    ax.set_axisbelow(True)

    path = os.path.join(out_dir, f"{meta['slug']}_C_spectrum.png")
    _save(fig, path)
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# D — Mass exponents  τ(q)
# ─────────────────────────────────────────────────────────────────────────────

def tau(dims: dict, meta: dict, out_dir: str):
    plt.rcParams.update(_DARK_RCPARAMS)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')

    q_plot, tau_plot, D0 = dims['q_plot'], dims['tau_plot'], dims['D0']

    if len(tau_plot) > 1:
        ax.fill_between(q_plot, tau_plot, alpha=0.12, color=TAU_COLOR, zorder=1)
        ax.plot(q_plot, tau_plot, '-', color=TAU_COLOR, alpha=0.25, lw=6, zorder=2)
        ax.plot(q_plot, tau_plot, 'o-', color=TAU_COLOR, ms=6, lw=2, zorder=3,
                markeredgecolor='white', markeredgewidth=0.5, label='τ(q) measured')
        if D0 > 0:
            tau_mono = (q_plot - 1) * D0
            ax.plot(q_plot, tau_mono, '--', color=MONO_COLOR, alpha=0.7, lw=1.8,
                    zorder=2, label=f'Monofractal ref (D={D0:.3f})')
            ax.fill_between(q_plot, tau_plot, tau_mono,
                            where=(tau_plot < tau_mono),
                            alpha=0.15, color=MONO_COLOR, zorder=1)
        ax.legend(fontsize=9, facecolor='#0f0f1a', edgecolor='#2a2a4a',
                  labelcolor=TEXT)

    ax.axhline(0, color='#444466', lw=0.8, ls='--')
    ax.axvline(0, color='#444466', lw=0.8, ls='--')
    ax.set_title(r"Mass Exponents  $\tau(q)$", fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel(r"$q$", fontsize=12, color=TEXT)
    ax.set_ylabel(r"$\tau(q)$", fontsize=12, color=TEXT)
    ax.tick_params(colors='#666688')
    ax.grid(True, color=GRID_COLOR, lw=0.8, ls='--')
    ax.set_axisbelow(True)

    path = os.path.join(out_dir, f"{meta['slug']}_D_tau.png")
    _save(fig, path)
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# E — f(α) vs q
# ─────────────────────────────────────────────────────────────────────────────

def fq(dims: dict, meta: dict, out_dir: str):
    plt.rcParams.update(_DARK_RCPARAMS)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')

    q_plot, f_plot = dims['q_plot'], dims['f_plot']

    if len(f_plot) > 1:
        ax.plot(q_plot, f_plot, '-', color='#aaaacc', alpha=0.5, lw=1.5, zorder=1)
        norm_q = Normalize(vmin=q_plot.min(), vmax=q_plot.max())
        ax.scatter(q_plot, f_plot, c=q_plot, cmap=plt.cm.plasma, norm=norm_q,
                   s=60, edgecolors='white', linewidths=0.5, zorder=2, alpha=0.95)
        ax.axvline(0, color='#444466', ls='--', lw=1.2, alpha=0.8)
        ax.text(0.02, 0.04, 'q=0\n(Total Footprint)',
                color='#8888aa', va='bottom', ha='left', fontsize=9,
                transform=ax.transAxes)
        cb = fig.colorbar(ScalarMappable(norm=norm_q, cmap=plt.cm.plasma),
                          ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('q value', color=TEXT, fontsize=10)
        cb.ax.yaxis.set_tick_params(color=TEXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
        cb.outline.set_edgecolor('#2a2a4a')

    ax.set_title(r"Subset Dimension vs Moment  $f(\alpha)$ vs $q$",
                 fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel(r"Moment  $q$", fontsize=12, color=TEXT)
    ax.set_ylabel(r"Fractal dimension  $f(\alpha)$", fontsize=12, color=TEXT)
    ax.tick_params(colors='#666688')
    ax.grid(True, color=GRID_COLOR, lw=0.8, ls='--')
    ax.set_axisbelow(True)

    path = os.path.join(out_dir, f"{meta['slug']}_E_fq.png")
    _save(fig, path)
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# F — Generalised dimensions  D(q)
# ─────────────────────────────────────────────────────────────────────────────

def Dq(dims: dict, meta: dict, out_dir: str):
    plt.rcParams.update(_DARK_RCPARAMS)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')

    q_plot, tau_plot = dims['q_plot'], dims['tau_plot']
    D0, D1, D2       = dims['D0'], dims['D1'], dims['D2']
    Dq_arr           = dims['Dq_arr']    # pre-built in dimensions.py — same source as D0/D1/D2

    if len(Dq_arr) > 1:
        valid = ~np.isnan(Dq_arr)

        # ── Tight y-axis so the curve fills the plot ──────────────────────────
        _d_vals  = Dq_arr[valid]
        _d_range = _d_vals.max() - _d_vals.min()
        _margin  = max(_d_range * 0.30, 0.05)
        ax.set_ylim(_d_vals.min() - _margin, _d_vals.max() + _margin * 1.8)

        ax.fill_between(q_plot[valid], Dq_arr[valid],
                        alpha=0.10, color=ACCENT2, zorder=1)
        ax.plot(q_plot[valid], Dq_arr[valid], '-',
                color=ACCENT2, alpha=0.30, lw=5, zorder=2)
        norm_q = Normalize(vmin=q_plot.min(), vmax=q_plot.max())
        ax.scatter(q_plot[valid], Dq_arr[valid],
                   c=q_plot[valid], cmap=plt.cm.plasma, norm=norm_q,
                   s=70, edgecolors='white', linewidths=0.5, zorder=3, alpha=0.95)

        ax.axhline(D0, ls='--', color=MONO_COLOR, lw=1.5, alpha=0.7,
                   label=f'Monofractal ref  D₀ = {D0:.3f}')

        # ── Text labels on each vertical marker ───────────────────────────────
        # d_mark is interpolated from the SAME Dq_arr used to draw the curve,
        # so the label number is guaranteed to match the visual intersection.
        _trans    = ax.get_xaxis_transform()
        _q_valid  = q_plot[valid]
        _dq_valid = Dq_arr[valid]
        for q_mark, label, col in [
            (0, '$D_0$', ACCENT1),
            (1, '$D_1$', ACCENT2),
            (2, '$D_2$', ACCENT3),
        ]:
            d_mark = float(np.interp(q_mark, _q_valid, _dq_valid))
            ax.axvline(q_mark, color=col, ls=':', lw=1.2, alpha=0.6)
            ax.text(q_mark + 0.08, 0.97, f'{label}\n{d_mark:.3f}',
                    transform=_trans,
                    color=col, fontsize=8.5, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.28', fc=PANEL_BG,
                              ec=col, lw=0.8, alpha=0.88))

        cb = fig.colorbar(ScalarMappable(norm=norm_q, cmap=plt.cm.plasma),
                          ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('q value', color=TEXT, fontsize=10)
        cb.ax.yaxis.set_tick_params(color=TEXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
        cb.outline.set_edgecolor('#2a2a4a')
        ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor='#2a2a4a',
                  labelcolor=TEXT)

    ax.set_title(r"Generalised Dimensions  $D(q)$ vs $q$",
                 fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel(r"Moment  $q$", fontsize=12, color=TEXT)
    ax.set_ylabel(r"$D(q) = \tau(q)\,/\,(q-1)$", fontsize=12, color=TEXT)
    ax.tick_params(colors='#666688')
    ax.grid(True, color=GRID_COLOR, lw=0.8, ls='--')
    ax.set_axisbelow(True)

    path = os.path.join(out_dir, f"{meta['slug']}_F_Dq.png")
    _save(fig, path)
    plt.rcParams.update(plt.rcParamsDefault)
