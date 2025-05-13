import json, jsonlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.colors import ListedColormap
import seaborn as sns
import os
import joblib

from sklearn.tree            import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, mean_squared_error, r2_score, auc

from retrieve_irtf import param_retrieve, get_spectra, set_spectra_name


def main():
    # Load parameters
    IDs, Teffs, loggs, Zs = param_retrieve()
    teff_logg_z = pd.DataFrame({
        'ID'  : IDs,
        'Teff': Teffs,
        'logg': loggs,
        'Z'   : Zs,
    })

    distances_name = 'used_distances'
    output_dir     = './SuitabilityPlots/'
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
            'mean_distance': np.mean(d[distances_name]),
            'max_distance': np.max(d[distances_name]),
        }
        records.append(rec)

    df = pd.DataFrame(records)

    print(df.describe(include='all')) # Show basic stats

    # Set plotting styles
    sns.set(style='whitegrid', font_scale=1.1)

    # ============================================================================
    # Function to re-run plotting for a subset
    def make_plots(sub_df, output_subdir, highlight=False):
        os.makedirs(output_subdir, exist_ok=True)

        # Kernel histogram
        plt.figure(figsize=(6,4))
        sns.countplot(x='kernel', data=sub_df, order=sub_df['kernel'].value_counts().index)
        plt.title('Kernel distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'kernel_histogram.png'))
        plt.close()

        # n_neighbours histogram
        plt.figure(figsize=(6,4))
        sns.histplot(sub_df['n_neighbours'], bins=np.arange(sub_df['n_neighbours'].min(), sub_df['n_neighbours'].max()+2)-0.5, kde=False)
        plt.title('Number of Neighbours distribution')
        plt.xlabel('n_neighbours')
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'n_neighbours_histogram.png'))
        plt.close()

        # Neighbour distances histogram
        all_distances = []
        for ID, d in data.items():
            if ID in sub_df['ID'].values:
                all_distances.extend(d[distances_name])


        plt.figure(figsize=(6,4))
        sns.histplot(all_distances, bins=30)
        plt.title('Neighbour Distances Distribution')
        plt.xlabel('Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'neighbour_distances_histogram.png'))
        plt.close()

        # Correlation plots
        def plot_with_histogram_density_subset(x, y, hue, title, ylabel, output_filename):
            kernels = sorted(sub_df[x].unique())
            neighbours = sorted(sub_df[hue].unique())

            fig, ax1 = plt.subplots(figsize=(12,6))
            ax2 = ax1.twinx()

            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.patch.set_visible(False)

            # Create a subset based on the desired parameter
            count_table   = sub_df.groupby([x, hue]).size().unstack(fill_value=0).reindex(index=kernels, columns=neighbours, fill_value=0)
            density_table = count_table.div(count_table.sum(axis=1), axis=0).fillna(0)

            width  = 0.5
            bottom = np.zeros(len(kernels))
            cmap   = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(len(neighbours))]

            for idx, n in enumerate(neighbours):
                densities = density_table[n].values
                ax2.bar(
                    range(len(kernels)), densities, width, bottom=bottom,
                    label=f'nn={n}', color=colors[idx], alpha=0.7
                )
                bottom += densities

            ax2.set_ylabel('Optimal n_neighbours (Normalised)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.set_ylim(0, 1)

            means = sub_df.groupby(x)[y].mean().reindex(kernels)
            ax1.plot(range(len(kernels)), means, marker='o', color='k', label=ylabel)
            ax1.set_xlabel('Kernel')
            ax1.set_ylabel(ylabel, color='k')
            ax1.tick_params(axis='y', labelcolor='k')

            ax1.set_xticks(range(len(kernels)))
            ax1.set_xticklabels(kernels, rotation=45, ha='right')

            lines_labels  = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            ax1.legend(lines, labels, loc='upper left', fontsize='small', bbox_to_anchor=(0.95, -0.05))
            plt.title(title)
            fig.tight_layout()

            plt.savefig(os.path.join(output_subdir, output_filename))
            plt.close()

        plot_with_histogram_density_subset(
            x='kernel', y='mse', hue='n_neighbours',
            title='MSE vs Kernel with kernel counts',
            ylabel='Mean MSE',
            output_filename='mse_vs_kernel_histogram.png'
        )

        plot_with_histogram_density_subset(
            x='kernel', y='max_distance', hue='n_neighbours',
            title='Max Distance vs Kernel with kernel counts',
            ylabel='Average Maximum Neighbour Distance',
            output_filename='maxdist_vs_kernel_histogram.png'
        )

        plot_with_histogram_density_subset(
            x='kernel', y='mean_distance', hue='n_neighbours',
            title='Mean Distance vs Kernel with kernel counts',
            ylabel='Average Neighbour Distance',
            output_filename='meandist_vs_kernel_histogram.png'
        )


        merged = pd.merge(teff_logg_z, sub_df[['ID', 'mse']], on='ID', how='left')

        # Scatter plot: Teff vs logg
        plt.figure(figsize=(8,6))
        plt.scatter(merged['Teff'], merged['logg'], c='grey', alpha=0.5, s=10, label='Subset stars')
        plt.xlabel('Teff (K)')
        plt.ylabel('logg')
        plt.title('Teff vs logg')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'teff_logg_mse.png'))
        plt.close()

        # Scatter plot: Teff vs Z
        plt.figure(figsize=(8,6))
        plt.scatter(merged['Teff'], merged['Z'], c='grey', alpha=0.5, s=10, label='Subset stars')
        plt.xlabel('Teff (K)')
        plt.ylabel('Z (metallicity)')
        plt.title('Teff vs Z')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'teff_z_mse.png'))
        plt.close()

        # MSE distribution
        plt.figure(figsize=(6,4))
        sns.histplot(sub_df['mse'], bins=30)
        plt.title('MSE Distribution')
        plt.xlabel('MSE')
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'mse_distribution.png'))
        plt.close()

        # Epsilon histogram
        plt.figure(figsize=(6,4))
        sns.countplot(x='epsilon', data=sub_df, order=sub_df['epsilon'].value_counts().index)
        plt.title('Epsilon distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'epsilon_histogram.png'))
        plt.close()

        # Smoothing histogram
        plt.figure(figsize=(6,4))
        sns.countplot(x='smoothing', data=sub_df, order=sub_df['smoothing'].value_counts().index)
        plt.title('Smoothing distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, 'smoothing_histogram.png'))
        plt.close()

    # ============================================================================
    # Now split and run

    above_01 = df[df['mse'] > 0.1]
    below_01 = df[df['mse'] < 0.1]

    make_plots(above_01, output_dir + '/Above_0.1')
    make_plots(below_01, output_dir + '/Below_0.1')
    make_plots(df, output_dir + '/All')

    print('Subset plots saved.')

    # ============================================================================
    # Analyse good fits with MSE < 0.02

    # Filter good fits
    good_fits = df[df['mse'] < 0.02]

    # Merge good fits with stellar parameters
    merged_good = pd.merge(teff_logg_z, good_fits, on='ID', how='inner')

    # Create output directory
    good_dir = os.path.join(output_dir, 'GoodFits')
    os.makedirs(good_dir, exist_ok=True)

    # Scatter plot helpers
    def scatter_param(ax, data, param, title, cmap='tab10'):
        unique_vals = sorted(data[param].dropna().unique())
        mapping = {val: i for i, val in enumerate(unique_vals)}
        colours = data[param].map(mapping)

        scatter = ax.scatter(data['Teff'], data['logg'], c=colours, cmap=cmap, alpha=0.8, s=20)
        ax.set_title(title)
        ax.set_xlabel('Teff (K)')
        ax.set_ylabel('logg')
        ax.invert_yaxis()

        # Custom legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=scatter.cmap(scatter.norm(i)),
                            markersize=8, label=str(label))
                for label, i in mapping.items()]
        ax.legend(handles=handles, title=param, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Plot each parameter
    params = ['smoothing', 'epsilon', 'kernel', 'n_neighbours']
    for param in params:
        fig, ax = plt.subplots(figsize=(7,6))
        scatter_param(ax, merged_good, param, f'Best {param} for MSE < 0.02')
        plt.tight_layout()
        plt.savefig(os.path.join(good_dir, f'best_{param}_vs_teff_logg.png'))
        plt.close()

    print('Good fit parameter plots saved.')

    # ============================================================================
    # Decision Tree Classifier for each parameter
    # ============================================================================

    ml_output_dir = 'ML_Parameter_Predictions'

    ml_dir = os.path.join(output_dir, ml_output_dir)
    os.makedirs(ml_dir, exist_ok=True)

    # Categorical vs Continuous params
    categorical_params = ['kernel']
    continuous_params  = ['smoothing', 'epsilon', 'n_neighbours']
    all_params         = categorical_params + continuous_params
    ml_dir             = os.path.join(output_dir, ml_output_dir)
    os.makedirs(ml_dir, exist_ok=True)

    label_encoders = {}

    for param in all_params:
        print(f'Training model for parameter: {param}')
        
        data = merged_good[['Teff', 'logg', 'Z', param]].dropna()
        X    = data[['Teff', 'logg', 'Z']]

        # Categorical parameters must be handled differently
        if param in categorical_params:
            y                     = data[param]
            le                    = LabelEncoder()
            y                     = le.fit_transform(y)
            label_encoders[param] = le
            model                 = DecisionTreeClassifier(max_depth=4, random_state=42)
        else:
            y     = data[param].astype(float)
            model = DecisionTreeRegressor(max_depth=4, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f'\nModel for {param}:')
        if param in categorical_params:
            print(classification_report(y_test, y_pred, target_names=le.classes_))
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            print(f'MSE: {mse:.4f}, R²: {r2:.4f}')

        # Plot tree
        fig, ax     = plt.subplots(figsize=(12, 6))
        class_names = le.classes_ if param in label_encoders else None
        plot_tree(model, feature_names=['Teff', 'logg', 'Z'], class_names=class_names, filled=True, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(ml_dir, f'decision_tree_{param}.pdf'), format='pdf')
        plt.close()

    # Prepare full isochrone grid
    isochrone_grid   = merged_good[['Teff', 'logg', 'Z']].dropna().drop_duplicates().reset_index(drop=True)
    predicted_params = isochrone_grid.copy()

    # Dictionary to store the trained models and metadata
    pretrained_model = {
        'models': {},             # Holds trained model objects
        'label_encoders': {},     # Holds label encoders for categorical params
        'features': ['Teff', 'logg', 'Z']
    }

    for param in all_params:
        print(f'Predicting {param} across isochrone...')
        X_all = isochrone_grid[['Teff', 'logg', 'Z']]

        # Load and train model
        if param in categorical_params:
            le = label_encoders[param]
            pretrained_model['label_encoders'][param] = le

            model = DecisionTreeClassifier(max_depth=4, random_state=42)
            data  = merged_good[['Teff', 'logg', 'Z', param]].dropna()
            y     = le.transform(data[param])
        else:
            model = DecisionTreeRegressor(max_depth=4, random_state=42)
            data  = merged_good[['Teff', 'logg', 'Z', param]].dropna()
            y     = data[param].astype(float)

        X = data[['Teff', 'logg', 'Z']]
        model.fit(X, y)
        pretrained_model['models'][param] = model

        # Predict on isochrone
        y_pred = model.predict(X_all)
        if param == 'n_neighbours':
            y_pred = np.array([min([4, 5, 6], key=lambda x: abs(x - y)) for y in y_pred])

        if param in categorical_params:
            y_pred = le.inverse_transform(np.round(y_pred).astype(int))

        predicted_params[param] = y_pred

    # Save predictions
    predicted_params.to_csv(os.path.join(output_dir, 'predicted_best_parameters.csv'), index=False)
    print("Predictions saved to predicted_best_parameters.csv")

    # Save pretrained model
    model_path = os.path.join('Stellar_Spectra', 'pretrained_interp_model.joblib')
    joblib.dump(pretrained_model, model_path)
    print(f"Pretrained interpolation model saved to {model_path}")


    # Setup plot style
    sns.set(style="whitegrid")
    plot_vars = [('Teff', 'logg'), ('Teff', 'Z'), ('logg', 'Z')]

    # Convert kernel to numeric for colourmapping
    predicted_params['kernel_num'] = pd.Categorical(predicted_params['kernel']).codes

    os.makedirs(os.path.join(output_dir, ml_output_dir), exist_ok=True)

    # Define kernel colours once
    unique_kernels   = predicted_params['kernel'].unique()
    kernel_colours   = sns.color_palette('tab10', n_colors=len(unique_kernels))
    kernel_to_colour = dict(zip(unique_kernels, kernel_colours))

    # Plot all parameters
    for param in all_params:
        fig, axes = plt.subplots(1, len(plot_vars), figsize=(6 * len(plot_vars), 5), constrained_layout=True)
        fig.suptitle(f"Predicted {param} across input space", fontsize=16)

        for ax, (xvar, yvar) in zip(axes, plot_vars):
            if param == 'kernel':
                colour_values = predicted_params['kernel'].map(kernel_to_colour)
                ax.scatter(
                    predicted_params[xvar], predicted_params[yvar],
                    c=colour_values.tolist(), edgecolor='k', s=50
                )
                ax.set_title(f"{xvar} vs {yvar}")
                ax.set_xlabel(xvar)
                ax.set_ylabel(yvar)
            else:
                scatter = ax.scatter(
                    predicted_params[xvar], predicted_params[yvar],
                    c=predicted_params[param], cmap='viridis',
                    edgecolor='k', s=50
                )
                ax.set_title(f"{xvar} vs {yvar}")
                ax.set_xlabel(xvar)
                ax.set_ylabel(yvar)
                if param == 'n_neighbours':
                    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', ticks=[4, 5, 6]) # Set to fixed integers - not continuous
                    cbar.set_ticklabels(['4', '5', '6'])
                else:
                    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
                cbar.set_label(param)

        # Kernel legend only on the full figure
        if param == 'kernel':
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=kernel, markerfacecolor=col, markersize=10) for kernel, col in kernel_to_colour.items()
            ]
            fig.legend(handles=handles, title='Kernel', loc='upper right')

        # Save figure
        fig.savefig(os.path.join(output_dir, ml_output_dir, f'MultiPanel_{param}.png'))
        plt.close(fig)

    # ============================================================================
    # Plot ROC curves #


    def plot_all_roc_curves():
        import matplotlib.pyplot as plt

        roc_file = './Stellar_Spectra/ROC.jsonl'
        entries  = []

        # Use jsonlines to load line-delimited JSON
        with jsonlines.open(roc_file) as reader:
            for entry in reader:
                entries.append(entry)

        # Plot each ROC curve
        plt.figure(figsize=(8, 6))
        for entry in entries:
            plt.plot(entry['fpr'], entry['tpr'], label=f"{entry['ID']} (AUC = {entry['roc_auc']:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('All ROC Curves')
        #plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'All_ROC_Curves.png'))
        plt.close()

    def plot_mean_roc_curve():
        roc_file = './Stellar_Spectra/ROC.jsonl'
        entries = []

        # Load using jsonlines
        with jsonlines.open(roc_file) as reader:
            for entry in reader:
                entries.append(entry)

        # Interpolate TPRs to a common FPR axis
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []

        for entry in entries:
            fpr        = np.array(entry['fpr'])
            tpr        = np.array(entry['tpr'])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        std_tpr  = np.std(interp_tprs, axis=0)
        mean_auc = np.mean([entry['roc_auc'] for entry in entries])
        std_auc  = np.std([entry['roc_auc'] for entry in entries])

        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', color='blue')
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2, label='±1 std. dev.')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'Mean_ROC_Curve.png'))
        plt.close()


    plot_all_roc_curves()
    plot_mean_roc_curve()

    #######################            # Use fixed categorical colours
    ###############################################
    ### Mean and max differences between interpolated and true spectra ###


    # Load actual spectra and their parameters
    param_vectors = np.vstack((Teffs, loggs, Zs)).T

    actual_spectra = {}
    wavelength = None
    for ID in IDs:
        try:
            spec               = get_spectra(ID)
            actual_spectra[ID] = spec[:, 1]
            if wavelength is None:
                wavelength = spec[:, 0]
        except Exception as e:
            print(f"Skipping ID {ID}: {e}")

    if len(actual_spectra) == 0:
        raise ValueError("No spectra were successfully loaded.")

    # Directory containing interpolated spectra
    spectra_dir        = './Stellar_Spectra'
    interpolated_files = [f for f in os.listdir(spectra_dir) if f.startswith("intspectra") and not f.endswith(".png")]

    all_mean_diffs    = []
    all_max_diffs     = []
    all_mses          = []
    subset_mean_diffs = []
    subset_max_diffs  = []
    subset_mses       = []

    for fname in interpolated_files:
        full_path = os.path.join(spectra_dir, fname)
        try:
            # Load interpolated spectrum (space-delimited, 2 columns)
            interp_spec = np.loadtxt(full_path)
            if interp_spec.shape[1] != 2:
                print(f"Skipping file {fname}: Not a 2-column file.")
                continue
        except Exception as e:
            print(f"Skipping file {fname}: {e}")
            continue

        # Extract parameters from filename
        try:
            # Example fname: intspectra_Teff4738_logg+4.39_Z+0.14
            parts = fname.replace("intspectra_", "").split("_")
            teff  = float(parts[0].replace("Teff", ""))
            logg  = float(parts[1].replace("logg", ""))
            Z     = float(parts[2].replace("Z", ""))
        except Exception as e:
            print(f"Skipping file {fname}: failed to parse parameters: {e}")
            continue

        # Find matching ID by parameters
        param       = np.array([teff, logg, Z])
        diffs       = np.linalg.norm(param_vectors - param, axis=1)
        idx         = np.argmin(diffs)
        ID          = IDs[idx]
        actual_spec = actual_spectra.get(ID)

        if actual_spec is None:
            print(f"No actual spectrum found for ID {ID}")
            continue
        if len(actual_spec) != interp_spec.shape[0]:
            print(f"Shape mismatch for {fname} vs ID {ID}")
            continue

        interp_flux = interp_spec[:, 1]
        diff        = np.abs(interp_flux - actual_spec)
        mean_diff   = np.mean(diff)
        max_diff    = np.max(diff)
        mse         = mean_squared_error(actual_spec, interp_flux)

        all_mean_diffs.append(mean_diff)
        all_max_diffs.append(max_diff)
        all_mses.append(mse)

        if mse < 0.1:
            subset_mean_diffs.append(mean_diff)
            subset_max_diffs.append(max_diff)
            subset_mses.append(mse)

    # Summary statistics with empty-list guards
    print("\n=== Summary Across All Spectra ===")
    if all_mean_diffs:
        print(f"Mean of Mean Differences: {np.mean(all_mean_diffs):.6f}")
        print(f"Max of Max Differences:   {np.max(all_max_diffs):.6f}")
        print(f"Average MSE:              {np.mean(all_mses):.6f}")
    else:
        print("No valid spectra were processed.")

    if subset_mean_diffs:
        print("\n=== Summary for Spectra with MSE < 0.1 ===")
        print(f"Mean of Mean Differences: {np.mean(subset_mean_diffs):.6f}")
        print(f"Max of Max Differences:   {np.max(subset_max_diffs):.6f}")
        print(f"Average MSE:              {np.mean(subset_mses):.6f}")
    else:
        print("\nNo spectra with MSE < 0.1.")

if __name__ == "__main__":
    main()