import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from retrieve_irtf import param_retrieve

output_dir = './SuitabilityPlots/'
os.makedirs(output_dir, exist_ok=True)

# Load saved JSON
with open('Stellar_Spectra/interpolation_results.json', 'r') as f:
    data = json.load(f)

# Build the DataFrame
records = []
for ID, d in data.items():
    rec = {
        'ID': ID,
        'epsilon': d['epsilon'],
        'smoothing': d['smoothing'],
        'kernel': d['kernel'],
        'n_neighbours': int(d['n_neighbours']),  # Explicitly convert to integer
        'mse': d['mse'],
        'mean_distance': np.mean(d['neighbour_distances']),
        'max_distance': np.max(d['neighbour_distances']),
    }
    records.append(rec)

df = pd.DataFrame(records)

# Ensure the column `n_neighbours` is of type int after DataFrame creation
df['n_neighbours'] = df['n_neighbours'].astype(int)

# Check data consistency by printing a small sample
print(df[['ID', 'n_neighbours']].head())

# Show basic stats to verify the data
print(df.describe(include='all'))

# Continue with the plots

# Set plotting styles
sns.set(style='whitegrid', font_scale=1.1)

# ============================================================================
# 1. Kernel histogram
plt.figure(figsize=(6,4))
sns.countplot(x='kernel', data=df, order=df['kernel'].value_counts().index)
plt.title('Kernel distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir+'/'+'kernel_histogram.png')
plt.close()

# ============================================================================
# 2. n_neighbours histogram
plt.figure(figsize=(6,4))
sns.histplot(df['n_neighbours'], bins=np.arange(df['n_neighbours'].min(), df['n_neighbours'].max()+2)-0.5, kde=False)
plt.title('Number of Neighbours distribution')
plt.xlabel('n_neighbours')
plt.tight_layout()
plt.savefig(output_dir+'/'+'n_neighbours_histogram.png')
plt.close()

# ============================================================================
# 3. Neighbour distances histogram
all_distances = []
for d in data.values():
    all_distances.extend(d['neighbour_distances'])

plt.figure(figsize=(6,4))
sns.histplot(all_distances, bins=30)
plt.title('Neighbour Distances Distribution')
plt.xlabel('Distance')
plt.tight_layout()
plt.savefig(output_dir+'/'+'neighbour_distances_histogram.png')
plt.close()

# ============================================================================
# 4. Correlation plots (new: Kernel on x-axis, coloured by n_neighbours)

def plot_with_histogram_density(x, y, hue, title, ylabel, output_filename):
    kernels = sorted(df[x].unique())
    neighbours = sorted(df[hue].unique())

    # Create figure and first y-axis
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Second y-axis
    ax2 = ax1.twinx()

    ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax1.patch.set_visible(False) # hide the 'canvas'

    # Prepare the data: counts of (kernel, n_neighbours)
    count_table = df.groupby([x, hue]).size().unstack(fill_value=0).reindex(index=kernels, columns=neighbours, fill_value=0)

    # Now normalise so each stacked bar sums to 1
    density_table = count_table.div(count_table.sum(axis=1), axis=0).fillna(0)

    width = 0.5
    bottom = np.zeros(len(kernels))

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(neighbours))]

    for idx, n in enumerate(neighbours):
        densities = density_table[n].values
        ax2.bar(
            range(len(kernels)), densities, width, bottom=bottom,
            label=f'n_neighbours={n}', color=colors[idx], alpha=0.7
        )
        bottom += densities

    ax2.set_ylabel('Optimal n_neighbours (Normalised)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 1)  # Ensure the y-axis always goes 0-1 for density

    # Line plot: mean y per kernel
    means = df.groupby(x)[y].mean().reindex(kernels)
    ax1.plot(range(len(kernels)), means, marker='o', color='k', label=ylabel)
    ax1.set_xlabel('Kernel')
    ax1.set_ylabel(ylabel, color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # Ensure x-ticks match the kernel values correctly
    ax1.set_xticks(range(len(kernels)))
    ax1.set_xticklabels(kernels, rotation=45, ha='right')

    # Legends (combine line and bar legends)
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels[0] = 'Avg Dist'
    labels = [label.replace('n_neighbours', 'nn') for label in labels]

    ax1.legend(lines, labels, loc='upper left', fontsize='small', bbox_to_anchor=(0.95, -0.05))
    # Titles and layout
    plt.title(title)
    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()

plot_with_histogram_density(
    x='kernel', y='mse', hue='n_neighbours',
    title='MSE vs Kernel with kernel counts',
    ylabel='Mean MSE',
    output_filename='mse_vs_kernel_histogram.png'
)

plot_with_histogram_density(
    x='kernel', y='max_distance', hue='n_neighbours',
    title='Max Distance vs Kernel with kernel counts',
    ylabel='Average Maximum Neighbour Distance',
    output_filename='maxdist_vs_kernel_histogram.png'
)

plot_with_histogram_density(
    x='kernel', y='mean_distance', hue='n_neighbours',
    title='Mean Distance vs Kernel with kernel counts',
    ylabel='Average Neighbour Distance',
    output_filename='meandist_vs_kernel_histogram.png'
)


# ============================================================================
# 5
# Load parameters
IDs, Teffs, loggs, Zs = param_retrieve()

# Build DataFrame
teff_logg_z = pd.DataFrame({
    'ID': IDs,
    'Teff': Teffs,
    'logg': loggs,
    'Z': Zs,
})

# Merge MSE
merged = pd.merge(teff_logg_z, df[['ID', 'mse']], on='ID', how='left')

# Mean MSE
mean_mse = df['mse'].mean()

# --- Plot Teff vs logg ---
plt.figure(figsize=(8,6))
plt.scatter(merged['Teff'], merged['logg'], c='grey', alpha=0.5, s=10, label='All stars')
plt.scatter(merged.loc[merged['mse'] > mean_mse, 'Teff'], 
            merged.loc[merged['mse'] > mean_mse, 'logg'], 
            c='orange', s=20, label='Above average MSE')
plt.scatter(merged.loc[merged['mse'] > mean_mse*1.2, 'Teff'], 
            merged.loc[merged['mse'] > mean_mse*1.2, 'logg'], 
            c='red', s=20, label='MSE > 120% of mean MSE')

plt.xlabel('Teff (K)')
plt.ylabel('logg')
plt.title('Teff vs logg, highlighting bad fits')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir+'/'+'teff_logg_mse.png')
plt.close()

# --- Plot Teff vs Z ---
plt.figure(figsize=(8,6))
plt.scatter(merged['Teff'], merged['Z'], c='grey', alpha=0.5, s=10, label='All stars')
plt.scatter(merged.loc[merged['mse'] > mean_mse, 'Teff'], 
            merged.loc[merged['mse'] > mean_mse, 'Z'], 
            c='orange', s=20, label='Above average MSE')
plt.scatter(merged.loc[merged['mse'] > mean_mse*1.2, 'Teff'], 
            merged.loc[merged['mse'] > mean_mse*1.2, 'Z'], 
            c='red', s=20, label='MSE > 120% of mean MSE')

plt.xlabel('Teff (K)')
plt.ylabel('Z (metallicity)')
plt.title('Teff vs Z, highlighting bad fits')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir+'/'+'teff_z_mse.png')
plt.close()

# ============================================================================
# 6. Additional analysis: MSE distribution

plt.figure(figsize=(6,4))
sns.histplot(df['mse'], bins=30)
plt.title('MSE Distribution')
plt.xlabel('MSE')
plt.tight_layout()
plt.savefig(output_dir+'/'+'mse_distribution.png')
plt.close()

# ============================================================================
print('Plots saved.')
