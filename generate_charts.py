"""
generate_charts.py
─────────────────────────────────────────────────────────────────────────────
EC7207 High Performance Computing | Group 02
Run this script from the project root directory:

    cd hpc-image-filtering
    python3 generate_charts.py

Charts are saved to:  report/figures/
Install dependencies: pip install matplotlib numpy
─────────────────────────────────────────────────────────────────────────────
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output directory — saved relative to wherever you run the script ──────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'report', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Saving figures to: {FIGURES_DIR}\n")

# ── Real benchmark data from terminal output ──────────────────────────────────
serial_gauss  = 0.113721
serial_sobel  = 0.078182

pt2_gauss  = 0.074347;  pt2_sobel  = 0.053266
pt4_gauss  = 0.057917;  pt4_sobel  = 0.049427

omp2_gauss = 0.095477;  omp2_sobel = 0.070498
omp4_gauss = 0.132910;  omp4_sobel = 0.061071

mpi2_gauss = 0.074185;  mpi2_sobel = 0.050564
mpi4_gauss = 0.051773;  mpi4_sobel = 0.040411

h2x2_gauss = 0.079316;  h2x2_sobel = 0.058481
h4x2_gauss = 0.086142;  h4x2_sobel = 0.073076

COLORS = {
    'serial'  : '#2C3E50',
    'pthreads': '#2980B9',
    'openmp'  : '#27AE60',
    'mpi'     : '#E67E22',
    'hybrid'  : '#8E44AD',
}

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
})

def save(name):
    plt.tight_layout()
    # Save both PNG (for quick viewing) and PDF (for LaTeX report)
    for ext in ['png', 'pdf']:
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {name}.png / {name}.pdf")

# Shared data used by multiple figures
labels = ['Serial\n(1)',
          'Pthreads\n(2T)', 'Pthreads\n(4T)',
          'OpenMP\n(2T)',   'OpenMP\n(4T)',
          'MPI\n(2P)',      'MPI\n(4P)',
          'Hybrid\n(2P×2T)','Hybrid\n(4P×2T)']

times_g = [serial_gauss,
           pt2_gauss, pt4_gauss,
           omp2_gauss, omp4_gauss,
           mpi2_gauss, mpi4_gauss,
           h2x2_gauss, h4x2_gauss]

times_s = [serial_sobel,
           pt2_sobel, pt4_sobel,
           omp2_sobel, omp4_sobel,
           mpi2_sobel, mpi4_sobel,
           h2x2_sobel, h4x2_sobel]

bar_colors = [COLORS['serial'],
              COLORS['pthreads'], COLORS['pthreads'],
              COLORS['openmp'],   COLORS['openmp'],
              COLORS['mpi'],      COLORS['mpi'],
              COLORS['hybrid'],   COLORS['hybrid']]

x = np.arange(len(labels))

legend_patches = [mpatches.Patch(color=COLORS[k], label=k.capitalize())
                  for k in ['serial', 'pthreads', 'openmp', 'mpi', 'hybrid']]

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Gaussian Blur Execution Time
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 1 — Gaussian Blur Execution Time...")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, times_g, color=bar_colors, width=0.6,
              edgecolor='white', linewidth=0.8)
for bar, t in zip(bars, times_g):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f'{t:.4f}', ha='center', va='bottom', fontsize=8)
ax.axhline(serial_gauss, color=COLORS['serial'], linestyle='--',
           linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Gaussian Blur — Execution Time Comparison')
ax.set_ylim(0, 0.180)
ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
save('fig1_gaussian_time')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Sobel Edge Detection Execution Time
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 2 — Sobel Edge Detection Execution Time...")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, times_s, color=bar_colors, width=0.6,
              edgecolor='white', linewidth=0.8)
for bar, t in zip(bars, times_s):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f'{t:.4f}', ha='center', va='bottom', fontsize=8)
ax.axhline(serial_sobel, color=COLORS['serial'], linestyle='--',
           linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Sobel Edge Detection — Execution Time Comparison')
ax.set_ylim(0, 0.115)
ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
save('fig2_sobel_time')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Gaussian Blur Speedup
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 3 — Gaussian Blur Speedup...")
speedups_g = [serial_gauss / t for t in times_g]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, speedups_g, color=bar_colors, width=0.6,
              edgecolor='white', linewidth=0.8)
for bar, s in zip(bars, speedups_g):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{s:.2f}x', ha='center', va='bottom', fontsize=8)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1,
           alpha=0.5, label='Baseline (1.00x)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Speedup  S = T_serial / T_parallel')
ax.set_title('Gaussian Blur — Speedup Comparison')
ax.set_ylim(0, 2.8)
ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
save('fig3_gaussian_speedup')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Sobel Edge Detection Speedup
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 4 — Sobel Edge Detection Speedup...")
speedups_s = [serial_sobel / t for t in times_s]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, speedups_s, color=bar_colors, width=0.6,
              edgecolor='white', linewidth=0.8)
for bar, s in zip(bars, speedups_s):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{s:.2f}x', ha='center', va='bottom', fontsize=8)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Speedup  S = T_serial / T_parallel')
ax.set_title('Sobel Edge Detection — Speedup Comparison')
ax.set_ylim(0, 2.5)
ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
save('fig4_sobel_speedup')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Scalability Line Chart (Pthreads vs OpenMP vs MPI, Gaussian)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 5 — Scalability Line Chart...")
fig, ax = plt.subplots(figsize=(7, 5))
workers  = [1, 2, 4]
pt_t  = [serial_gauss, pt2_gauss,  pt4_gauss]
omp_t = [serial_gauss, omp2_gauss, omp4_gauss]
mpi_t = [serial_gauss, mpi2_gauss, mpi4_gauss]

ax.plot(workers, pt_t,  'o-', color=COLORS['pthreads'],
        linewidth=2, markersize=7, label='Pthreads')
ax.plot(workers, omp_t, 's-', color=COLORS['openmp'],
        linewidth=2, markersize=7, label='OpenMP')
ax.plot(workers, mpi_t, '^-', color=COLORS['mpi'],
        linewidth=2, markersize=7, label='MPI')
ax.axhline(serial_gauss, color=COLORS['serial'], linestyle='--',
           linewidth=1.5, alpha=0.6, label='Serial baseline')

offsets = {'pt': (5, 6), 'omp': (5, -14), 'mpi': (5, 6)}
for w, t in zip(workers, pt_t):
    ax.annotate(f'{t:.4f}', (w, t), textcoords='offset points',
                xytext=offsets['pt'], fontsize=8, color=COLORS['pthreads'])
for w, t in zip(workers, omp_t):
    ax.annotate(f'{t:.4f}', (w, t), textcoords='offset points',
                xytext=offsets['omp'], fontsize=8, color=COLORS['openmp'])
for w, t in zip(workers, mpi_t):
    ax.annotate(f'{t:.4f}', (w, t), textcoords='offset points',
                xytext=offsets['mpi'], fontsize=8, color=COLORS['mpi'])

ax.set_xlabel('Number of Threads / Processes')
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Gaussian Blur — Scalability')
ax.set_xticks([1, 2, 4])
ax.set_ylim(0, 0.168)
ax.legend(fontsize=10)
save('fig5_scalability')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Parallel Efficiency (Gaussian Blur)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 6 — Parallel Efficiency...")
eff_workers = [2, 4, 2, 4, 2, 4, 4, 8]
eff_times   = [pt2_gauss,  pt4_gauss,
               omp2_gauss, omp4_gauss,
               mpi2_gauss, mpi4_gauss,
               h2x2_gauss, h4x2_gauss]
eff_colors  = [COLORS['pthreads'], COLORS['pthreads'],
               COLORS['openmp'],   COLORS['openmp'],
               COLORS['mpi'],      COLORS['mpi'],
               COLORS['hybrid'],   COLORS['hybrid']]
eff_labels  = ['Pthreads\n2T', 'Pthreads\n4T',
               'OpenMP\n2T',   'OpenMP\n4T',
               'MPI\n2P',      'MPI\n4P',
               'Hybrid\n2P×2T','Hybrid\n4P×2T']

efficiencies = [(serial_gauss / t) / n * 100
                for t, n in zip(eff_times, eff_workers)]

xe = np.arange(len(eff_labels))
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(xe, efficiencies, color=eff_colors, width=0.6,
              edgecolor='white', linewidth=0.8)
for bar, e in zip(bars, efficiencies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{e:.1f}%', ha='center', va='bottom', fontsize=9)
ax.axhline(100, color='gray', linestyle='--', linewidth=1,
           alpha=0.5, label='Ideal (100%)')
ax.set_xticks(xe)
ax.set_xticklabels(eff_labels)
ax.set_ylabel('Efficiency  E = (S / N) x 100%')
ax.set_title('Gaussian Blur — Parallel Efficiency')
ax.set_ylim(0, 125)
ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
save('fig6_efficiency')

# ─────────────────────────────────────────────────────────────────────────────
print("\nAll 6 figures generated successfully!")
print(f"Location: {FIGURES_DIR}")
print("\nFiles created:")
for f in sorted(os.listdir(FIGURES_DIR)):
    size = os.path.getsize(os.path.join(FIGURES_DIR, f))
    print(f"  {f:40s}  {size/1024:.1f} KB")