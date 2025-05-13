import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from itertools import combinations

visual       = False # Show plots interactively
heatmaperror = False # Show errors on heatmap

# Set up output
ouptut_dir = "SSP_Spectra/ComparisonPlots/"
if not os.path.exists(ouptut_dir): os.makedirs(ouptut_dir)

# Define paths to models
my_model_path   = "SSP_Spectra/SSP_Fe+0.281_a+0.0_C+0.0_N+0.0_O+0.0_Mg+0.0_Si+0.0_Ca+0.0_Ti+0.0_Na+0.0_Al+0.0_Ba+0.0_Eu+0.0_age10.0_kroupa"
old_model_path  = "../miolnir/SSP_Spectra/SSP_Fe+0.281_a+0.0_C+0.0_N+0.0_O+0.0_Mg+0.0_Si+0.0_Ca+0.0_Ti+0.0_Na+0.0_Al+0.0_Ba+0.0_Eu+0.0_age10.0_kroupa"
model_mars_path = './DATA/MarS/MARv_SAL_sed_NOCS_H_Z_0.029999999_Tg_1.0000000e+10'
model_girs_path = './DATA/GirS/GIRv_SAL_sed_NOCS_H_Z_0.029999999_Tg_1.0000000e+10'
model_bass_path = './DATA/BaSS/BASv_SAL_sed_NOCS_H_Z_0.029999999_Tg_1.0000000e+10'

# Load models (basic 2-column format)
def load_model(path):
    try:
        data = np.loadtxt(path)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None

# Load all models
new = np.loadtxt(my_model_path)
old  = np.loadtxt(old_model_path)
mars = np.loadtxt(model_mars_path)
girs = np.loadtxt(model_girs_path)
bass = np.loadtxt(model_bass_path)

x_new, y_new = new[:, 0], new[:, 1]
x_old,  y_old  = old[:, 0],  old[:, 1]
x_mars, y_mars = mars[:, 0]*10000, mars[:, 1]
x_girs, y_girs = girs[:, 0]*10000, girs[:, 1]
x_bass, y_bass = bass[:, 0]*10000, bass[:, 1]

# Information for telluric gap and normalisation
telluric_xl     = (18050-9350)+150
telluric_xr     = (18800-9350)+150
norm_wavelength = 12230



### Plot basic comparison of models ###  

# Set up plots
ax   = plt.subplot(111)
lwid = 0.3
font = 12
plt.xlabel(r'$\lambda (\AA)$', fontsize = font)
plt.ylabel(f'$F/\lambda_{{12230}}$', fontsize = font)
ax.set_title("SSP Model Comparison: Z=0.03, Age=10 Gyr")

inter                             = interp1d(x_mars,y_mars)
norm                              = inter(norm_wavelength)
y_mars[telluric_xl : telluric_xr] = np.nan
ax.plot(x_mars, y_mars/norm, 'r', linewidth = lwid, label = 'MarS Model')
er_mars = y_mars/norm

inter                             = interp1d(x_girs, y_girs)
norm                              = inter(norm_wavelength)
y_girs[telluric_xl : telluric_xr] = np.nan
ax.plot(x_girs, y_girs/norm, 'g', linewidth = lwid, label = 'GirS Model')
er_girs = y_girs/norm

inter                             = interp1d(x_bass, y_bass)
norm                              = inter(norm_wavelength)
y_bass[telluric_xl : telluric_xr] = np.nan
ax.plot(x_bass, y_bass/norm, 'm', linewidth = lwid, label = 'BaSS Model')
er_bass = y_bass/norm

inter                             = interp1d(x_new, y_new)
norm                              = inter(norm_wavelength)
y_new[telluric_xl : telluric_xr] = np.nan
ax.plot(x_new, y_new/norm, 'b--', linewidth = lwid, label = 'New Model')
er_new = y_new/norm

inter                            = interp1d(x_old, y_old)
norm                             = inter(norm_wavelength)
y_old[telluric_xl : telluric_xr] = np.nan
ax.plot(x_old, y_old/norm, '--', color='grey', linewidth = lwid, label = 'Old Model')
er_old = y_old/norm

ax.legend()

if visual: plt.show()
plot_path = os.path.join(ouptut_dir, "SSP_Model_Comparison.png")
plt.savefig(plot_path, dpi=600)
plt.close()



### Compute residuals and summary statistics ###

# Build shared wavelength grid (interpolate all onto x_new)
grid = x_new

interp_mars = interp1d(x_mars, er_mars, bounds_error=False, fill_value=np.nan)
interp_girs = interp1d(x_girs, er_girs, bounds_error=False, fill_value=np.nan)
interp_bass = interp1d(x_bass, er_bass, bounds_error=False, fill_value=np.nan)

mars_interp = interp_mars(grid)
girs_interp = interp_girs(grid)
bass_interp = interp_bass(grid)

# Compute average of literature models
lit_avg = np.nanmean(np.vstack([mars_interp, girs_interp, bass_interp]), axis=0)

# Interpolated versions of my new and old
interp_new = interp1d(x_new, er_new, bounds_error=False, fill_value=np.nan)
interp_old = interp1d(x_old,   er_old, bounds_error=False, fill_value=np.nan)

new_interp          = interp_new(grid)
old_interp          = interp_old(grid)
residual_new_vs_lit = new_interp - lit_avg
residual_new_vs_old = old_interp  - lit_avg

# Plot residuals
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(grid, residual_new_vs_lit,  'b',   linewidth=lwid, label='New Model - Literature Avg')
ax.plot(grid, residual_new_vs_old,  'k--', linewidth=lwid, label='Old Model - Literature Avg')
ax.axhline(0, color='grey', linestyle=':')
ax.set_xlabel(r'$\lambda (\AA)$')
ax.set_ylabel('Residual')
ax.set_title('Residuals of SSP Models')
ax.legend()
ax.set_xlim(left=grid[0], right=grid[-1])
plt.tight_layout()

if visual: plt.show()
plot_path = os.path.join(ouptut_dir, "AverageResiduals.png")
plt.savefig(plot_path, dpi=600)
plt.close()

# Print statistics
if visual:
    print(f"[New vs Literature Avg] Mean diff: {np.nanmean(residual_new_vs_lit):.4e}, Std: {np.nanstd(residual_new_vs_lit):.4e}")
    print(f"[New vs Old Model]      Mean diff: {np.nanmean(residual_new_vs_old):.4e}, Std: {np.nanstd(residual_new_vs_old):.4e}")


### Pairwise residuals between literature models ###
lit_models = {
    'MarS': mars_interp,
    'GirS': girs_interp,
    'BaSS': bass_interp
}

# Generate all pairs
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axes = axes.flatten()
for ax, (name1, name2) in zip(axes, combinations(lit_models, 2)):
    res       = lit_models[name1] - lit_models[name2]
    mean_diff = np.nanmean(res)
    std_diff  = np.nanstd(res)
    min_diff  = np.nanmin(res)
    max_diff  = np.nanmax(res)

    ax.plot(grid, res, label=f"{name1} - {name2}", linewidth=lwid)
    ax.axhline(0, color='grey', linestyle=':')
    ax.set_title(f"{name1} vs {name2}")
    ax.set_xlabel(r'$\lambda (\AA)$')
    ax.set_ylabel('Residual')

    textstr = f"mean: {mean_diff:.2e}\nstd: {std_diff:.2e}\nmin: {min_diff:.2e}\nmax: {max_diff:.2e}"
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize            = 9,
            verticalalignment   = 'bottom',
            horizontalalignment = 'right',
            bbox                = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.legend()
    print(f"[{name1} vs {name2}] Mean diff: {mean_diff:.4e}, Std: {std_diff:.4e}")


plt.tight_layout()

if visual: plt.show()
plot_path = os.path.join(ouptut_dir, "SSP_LiteratureComparisons.png")
plt.savefig(plot_path, dpi=600)
plt.close()

### Residuals between each literature model, and the new and old models ###
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for i, (name, model_interp) in enumerate(lit_models.items()):
    res_new = new_interp - model_interp
    res_old = old_interp - model_interp

    mean_new = np.nanmean(res_new)
    std_new  = np.nanstd(res_new)
    min_new  = np.nanmin(res_new)
    max_new  = np.nanmax(res_new)

    mean_old = np.nanmean(res_old)
    std_old  = np.nanstd(res_old)
    min_old  = np.nanmin(res_old)
    max_old  = np.nanmax(res_old)

    ax1 = axes[i]
    ax1.plot(grid, res_new, 'b', label=f'New - {name}', linewidth=lwid)
    ax1.axhline(0, color='grey', linestyle=':')
    ax1.set_title(f'New vs {name}')
    ax1.set_xlabel(r'$\lambda (\AA)$')
    ax1.set_ylabel('Residual')
    ax1.text(0.98, 0.02,
             f"mean: {mean_new:.2e}\nstd: {std_new:.2e}\nmin: {min_new:.2e}\nmax: {max_new:.2e}",
             transform           = ax1.transAxes,
             fontsize            = 9,
             verticalalignment   = 'bottom',
             horizontalalignment = 'right',
             bbox                = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    ax1.legend()
    print(f"[New vs {name}] Mean diff: {mean_new:.4e}, Std: {std_new:.4e}")

    ax2 = axes[i + 3]
    ax2.plot(grid, res_old, 'k--', label=f'Old - {name}', linewidth=lwid)
    ax2.axhline(0, color='grey', linestyle=':')
    ax2.set_title(f'Old vs {name}')
    ax2.set_xlabel(r'$\lambda (\AA)$')
    ax2.set_ylabel('Residual')

    ax2.text(0.98, 0.02,
             f"mean: {mean_old:.2e}\nstd: {std_old:.2e}\nmin: {min_old:.2e}\nmax: {max_old:.2e}",
             transform           = ax2.transAxes,
             fontsize            = 9,
             verticalalignment   = 'bottom',
             horizontalalignment = 'right',
             bbox                = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    ax2.legend()
    print(f"[Old vs {name}] Mean diff: {mean_old:.4e}, Std: {std_old:.4e}")


plt.tight_layout()

if visual: plt.show()
plot_path = os.path.join(ouptut_dir, "SSP_ModelComparisons.png")
plt.savefig(plot_path, dpi=600)
plt.close()

### Pairwise residual heatmaps with errors ###

# Prepare interpolated models on shared grid
all_models = {
    'new' : new_interp,
    'old' : old_interp
}
all_models  = lit_models | all_models
model_names = list(all_models.keys())
n_models    = len(model_names)

# Initialise matrices
mean_residuals         = np.full((n_models, n_models), np.nan)
mean_abs_residuals     = np.full((n_models, n_models), np.nan)
mean_residuals_err     = np.full((n_models, n_models), np.nan)
mean_abs_residuals_err = np.full((n_models, n_models), np.nan)

# Compute full matrix with errors
for i in range(n_models):
    for j in range(n_models):
        res        = all_models[model_names[i]] - all_models[model_names[j]]
        finite_res = res[np.isfinite(res)]

        if finite_res.size == 0: continue

        mean_residuals[i, j]     = np.nanmean(finite_res)
        mean_abs_residuals[i, j] = np.nanmean(np.abs(finite_res))

        # Use (n - 1) for comparisons that don't include self (i != j), else n
        effective_n = finite_res.size
        if i == j: divisor = np.sqrt(effective_n)
        else:      divisor = np.sqrt(effective_n - 1) if effective_n > 1 else np.nan

        # Standard error of the mean
        mean_residuals_err[i, j]     = np.nanstd(finite_res, ddof=1) / divisor
        mean_abs_residuals_err[i, j] = np.nanstd(np.abs(finite_res), ddof=1) / divisor


# Convert to DataFrame for seaborn and clustering
mean_residuals_df     = pd.DataFrame(mean_residuals,     index=model_names, columns=model_names)
mean_abs_residuals_df = pd.DataFrame(mean_abs_residuals, index=model_names, columns=model_names)

# Compute clustering order
dists         = squareform(mean_abs_residuals)
link          = linkage(dists, method='average')
order         = leaves_list(link)
ordered_names = [model_names[i] for i in order]

# Reorder matrices
mean_residuals_df       = mean_residuals_df.loc[ordered_names, ordered_names]
mean_abs_residuals_df   = mean_abs_residuals_df.loc[ordered_names, ordered_names]
mean_residuals_err      = pd.DataFrame(mean_residuals_err, index=model_names, columns=model_names).loc[ordered_names, ordered_names]
mean_abs_residuals_err  = pd.DataFrame(mean_abs_residuals_err, index=model_names, columns=model_names).loc[ordered_names, ordered_names]

### Signed Mean Residuals Heatmap ###
annot_residuals = mean_residuals_df.copy().astype(str)
for i in range(n_models):
    for j in range(n_models):
        val  = mean_residuals_df.iloc[i, j]
        err  = mean_residuals_err.iloc[i, j]
        if np.isfinite(val) and np.isfinite(err):
            annot_residuals.iloc[i, j] = f"{val:.2e}\pm{err:.1e}" if heatmaperror else f"{val:.2e}"
        else:
            annot_residuals.iloc[i, j] = ""

plt.figure(figsize=(8, 6))
sns.heatmap(mean_residuals_df, annot=annot_residuals, fmt="", cmap="coolwarm", cbar_kws={"label": "Mean Residual"})
plt.title("Pairwise Mean Residuals (Signed) between SSP Models")
plt.tight_layout()

if visual: plt.show()
plt.savefig(os.path.join(ouptut_dir, "SSP_ModelResiduals_Signed.png"), dpi=600)
plt.close()

### Mean Absolute Residuals Heatmap ###

# Only show lower diagonal, remove top row and last column
mask                             = np.zeros_like(mean_abs_residuals_df, dtype=bool)
mask[np.triu_indices_from(mask)] = True
reduced_df                       = mean_abs_residuals_df.iloc[1:, :-1]
reduced_err                      = mean_abs_residuals_err.iloc[1:, :-1]
reduced_mask                     = mask[1:, :-1]

# Create annotated heatmap with errors
annot_absres = reduced_df.copy().astype(str)
for i in range(reduced_df.shape[0]):
    for j in range(reduced_df.shape[1]):
        val = reduced_df.iloc[i, j]
        err = reduced_err.iloc[i, j]
        if np.isfinite(val) and np.isfinite(err):
            annot_absres.iloc[i, j] = f"{val:.2e}\pm{err:.1e}" if heatmaperror else f"{val:.2e}"
        else:
            annot_absres.iloc[i, j] = ""

plt.figure(figsize=(8, 6))
sns.heatmap(reduced_df, annot=annot_absres, fmt="", cmap="viridis", cbar_kws={"label": "Mean Abs. Residual"}, mask=reduced_mask)
plt.title("Pairwise Mean Absolute Residuals between SSP Models")
plt.tight_layout()

if visual: plt.show()
plt.savefig(os.path.join(ouptut_dir, "SSP_ModelResiduals_Abs.png"), dpi=600)
plt.close()

### Bar Chart: Average Absolute Deviation vs Literature Models ###

# Calculate average and error of absolute deviation from literature models
avg_abs_deviation_lit     = {}
avg_abs_deviation_lit_err = {}

for name in model_names:
    if name in lit_models:
        others = [mean_abs_residuals_df.loc[name, other] for other in lit_models if other != name]
    else:
        others = [mean_abs_residuals_df.loc[name, other] for other in lit_models]

    others = np.array(others)
    others = others[np.isfinite(others)]

    avg_abs_deviation_lit[name]     = np.nanmean(others)
    avg_abs_deviation_lit_err[name] = np.nanstd(others, ddof=1) / np.sqrt(len(others)) if len(others) > 0 else np.nan

# Sort by deviation for ranked plotting
sorted_items = sorted(avg_abs_deviation_lit.items(), key=lambda item: item[1])
sorted_names = [item[0] for item in sorted_items]
sorted_vals  = [item[1] for item in sorted_items]
sorted_errs  = [avg_abs_deviation_lit_err[name] for name in sorted_names]

plt.figure(figsize=(8, 5))
bars = plt.barh(sorted_names, sorted_vals, color='skyblue', xerr=sorted_errs, capsize=4, ecolor='grey')
plt.xlabel("Avg. Abs. Residuals from Literature Models")
plt.title("Model Difference from Literature")
plt.tight_layout()

# Annotate bars with residual error
for bar, val, err in zip(bars, sorted_vals, sorted_errs):
    y_pos = bar.get_y() + bar.get_height() / 2 + bar.get_height()*0.2
    plt.text(0.0005, y_pos, f"{val:.2e}±{err:.1e}", va='center', fontsize=9)

if visual: plt.show()
plt.savefig(os.path.join(ouptut_dir, "SSP_ModelDeviationFromLiterature.png"), dpi=600)
plt.close()
