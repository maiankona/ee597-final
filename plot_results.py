"""
plot_results.py — Generate all figures for the AdaSpray paper.
Run after adaspray.py has produced results.json.

Produces:
  fig1_delay.pdf         — Mean delivery delay vs encounter rate
  fig2_overhead.pdf      — Mean transmissions vs encounter rate
  fig3_delivery_ratio.pdf— Delivery ratio vs encounter rate
  fig4_k_adaptation.pdf  — AdaSpray K adaptation over time (single trial)
"""

import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (5.5, 3.8),
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

# ── Load results ───────────────────────────────────────────────────────────────
with open('results.json') as f:
    summary = json.load(f)

RATES = sorted(float(r) for r in summary['hold'].keys())

PROTO_STYLE = {
    'hold':      dict(label='Hold & Deliver', color='#888888', marker='s', ls='--'),
    'epidemic':  dict(label='Epidemic',       color='#d62728', marker='^', ls='--'),
    'adaspray':  dict(label='AdaSpray (ours)',color='#1f77b4', marker='o', ls='-', lw=2.4),
    'fixed_K2':  dict(label='Fixed K=2',      color='#2ca02c', marker='D', ls='-.'),
    'fixed_K4':  dict(label='Fixed K=4',      color='#ff7f0e', marker='v', ls='-.'),
    'fixed_K8':  dict(label='Fixed K=8',      color='#9467bd', marker='P', ls='-.'),
}

PROTOS = ['hold', 'epidemic', 'adaspray', 'fixed_K2', 'fixed_K4', 'fixed_K8']


def get(proto, metric):
    return [summary[proto][str(r)][metric] for r in RATES]

def get_err(proto, metric):
    # Return 95% CI = 1.96 * std / sqrt(n)
    vals = []
    for r in RATES:
        d = summary[proto][str(r)]
        std = d.get(f'std_{metric.replace("mean_","")}', 0)
        n = d.get('n_trials', 30)
        vals.append(1.96 * std / math.sqrt(n))
    return vals


# ── Figure 1: Mean Delivery Delay ─────────────────────────────────────────────
fig, ax = plt.subplots()
for p in PROTOS:
    y = get(p, 'mean_delay')
    yerr = get_err(p, 'std_delay')
    s = PROTO_STYLE[p]
    ax.errorbar(RATES, y, yerr=None,
                label=s['label'], color=s['color'],
                marker=s['marker'], ls=s['ls'],
                lw=s.get('lw', plt.rcParams['lines.linewidth']))

ax.set_xlabel('Encounter Rate (encounters/time unit)')
ax.set_ylabel('Mean Delivery Delay (time units)')
ax.set_title('Delivery Delay vs. Encounter Rate')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('fig1_delay.pdf', dpi=300)
print("Saved fig1_delay.pdf")


# ── Figure 2: Mean Transmissions (Overhead) ────────────────────────────────────
fig, ax = plt.subplots()
for p in PROTOS:
    y = get(p, 'mean_tx')
    s = PROTO_STYLE[p]
    ax.plot(RATES, y,
            label=s['label'], color=s['color'],
            marker=s['marker'], ls=s['ls'],
            lw=s.get('lw', plt.rcParams['lines.linewidth']))

ax.set_xlabel('Encounter Rate (encounters/time unit)')
ax.set_ylabel('Mean Number of Transmissions')
ax.set_title('Overhead vs. Encounter Rate')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('fig2_overhead.pdf', dpi=300)
print("Saved fig2_overhead.pdf")


# ── Figure 3: Delivery Ratio ───────────────────────────────────────────────────
fig, ax = plt.subplots()
for p in PROTOS:
    y = get(p, 'delivery_ratio')
    s = PROTO_STYLE[p]
    ax.plot(RATES, y,
            label=s['label'], color=s['color'],
            marker=s['marker'], ls=s['ls'],
            lw=s.get('lw', plt.rcParams['lines.linewidth']))

ax.set_xlabel('Encounter Rate (encounters/time unit)')
ax.set_ylabel('Delivery Ratio')
ax.set_title('Delivery Ratio vs. Encounter Rate')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('fig3_delivery_ratio.pdf', dpi=300)
print("Saved fig3_delivery_ratio.pdf")


# ── Figure 4: K Adaptation Over Time ──────────────────────────────────────────
# Re-run a single AdaSpray trial at two rates to show K changing
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from adaspray import run_trial

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, rate, title in [
    (axes[0], 0.2, 'Sparse Network (rate=0.2)'),
    (axes[1], 4.0, 'Dense Network (rate=4.0)'),
]:
    r = run_trial('adaspray', n_nodes=10, encounter_rate=rate,
                  duration=150.0, K_base=4, lambda_ref=1.0, alpha=0.3, seed=42)
    times = [x[0] for x in r['k_history']]
    ks    = [x[1] for x in r['k_history']]
    ax.step(times, ks, color='#1f77b4', lw=1.8, where='post')
    ax.axhline(4, color='gray', ls='--', lw=1.2, label='K_base=4')
    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Adaptive K')
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig.suptitle('AdaSpray: K Adaptation Over Time', fontsize=12)
fig.tight_layout()
fig.savefig('fig4_k_adaptation.pdf', dpi=300)
print("Saved fig4_k_adaptation.pdf")

print("\nAll figures generated.")
