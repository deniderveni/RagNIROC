import numpy as np
import matplotlib.pyplot as plt
import os
import json, jsonlines

from scipy.interpolate      import RBFInterpolator
from scipy.spatial          import distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, roc_curve, auc
from sklearn.neighbors      import NearestNeighbors

from concurrent.futures import ProcessPoolExecutor, as_completed

from retrieve_irtf      import param_retrieve, get_spectra, set_spectra_name

epsilons   = np.linspace(0.5, 5.0, 10)
smoothings = np.logspace(-1, 0, 5)
kernels    = ['gaussian', 'multiquadric'] # 'cubic' , 'inverse_multiquadric', 'linear', 'thin_plate_spline',
min_neigh  = 4
max_neigh  = 6

def evaluate_error(y_true, y_pred):
    """ Mean Squared Error (MSE) function """

    return np.mean((y_true - y_pred) ** 2)

def interpolate_one_star(i, param, ID, true_spectra, indices_list, distances_list, good_IDs, spectra, scaler, wavelength, epsilons, smoothings, kernels, min_neigh, good_params):
    """Interpolate a single star's spectrum using RBF interpolation."""

    # Load the parameters and spectra for the star
    try:
        target_param_scaled = scaler.transform([param])

        indices   = indices_list[i]
        distances = distances_list[i]

        if len(indices) < min_neigh:
            print(f"Skipping {ID}: not enough neighbours")
            return None

        best_mse      = float('inf')
        best_settings = {}

        # Itterate over all parameters
        for n_neigh in range(min_neigh, len(indices) + 1):
            neighbour_params  = good_params[indices[:n_neigh]]
            neighbour_spectra = spectra[indices[:n_neigh]]

            for epsilon in epsilons:
                for smoothing in smoothings:
                    for kernel in kernels:

                        # In these kernel cases, effect of epsilon and smoothing is identical according to documentation, so skip
                        if kernel in ["linear", "thin_plate_spline", "cubic", "quintic"] and epsilon != 1: continue                  
                    
                        # Apply RBF interpolation
                        try:
                            model = RBFInterpolator(neighbour_params, neighbour_spectra,
                                                    kernel=kernel,
                                                    epsilon=epsilon,
                                                    smoothing=smoothing)
                            int_spectra = model(target_param_scaled)[0]

                            mse = evaluate_error(true_spectra, int_spectra)

                            # Save settings for lowest MSE
                            if mse < best_mse:
                                best_mse = mse
                                best_settings = {
                                    'epsilon'       : epsilon,
                                    'smoothing'     : smoothing,
                                    'kernel'        : kernel,
                                    'mse'           : mse,
                                    'n_neighbours'  : n_neigh,
                                    'used_distances': distances[:n_neigh].tolist(),
                                    'used_IDs'      : [good_IDs[j] for j in indices[:n_neigh].tolist()]
                                }
                        except Exception:
                            continue

        if not best_settings:
            print(f"Skipping star {ID} due to interpolation failure")
            return None

        # Refit with best settings
        # TODO: Save the best model as well to avoid refitting step

        neighbour_params  = good_params[indices[:best_settings['n_neighbours']]]
        neighbour_spectra = spectra[indices[:best_settings['n_neighbours']]]

        model = RBFInterpolator(neighbour_params, neighbour_spectra,
                                kernel=best_settings['kernel'],
                                epsilon=best_settings['epsilon'],
                                smoothing=best_settings['smoothing'])
        int_spectra = model(target_param_scaled)[0]

        # Calculate ROC curve data
        fpr, tpr, _ = roc_curve(true_spectra > 0.01, int_spectra)  # using threshold on true_spectra > 0.01
        roc_auc     = auc(fpr, tpr)

        # Save ROC data
        roc_data = {
            'ID': ID,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_auc': roc_auc
        }

        # Save ROC data to file
        # TODO: Structure is poor, and creates a jsonl file realistically. Needed to fix to get it to work, fix this internally here
        with open('./Stellar_Spectra/ROC.json', 'a') as f:
            json.dump(roc_data, f, indent=4)
            f.write('\n')  # Write each entry on a new line

        # Save the interpolated spectrum
        output   = np.column_stack((wavelength, int_spectra))
        filename = set_spectra_name(param[0], param[1], param[2])
        txt_path = './Stellar_Spectra/' + filename
        png_path = txt_path + '.png'

        np.savetxt(txt_path, output)

        # Plot the interpolated spectrum
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(wavelength, int_spectra, 'b', label='Interpolated Spectrum', linewidth=0.5)
        ax.plot(wavelength, true_spectra, 'r--', label='True Spectrum', linewidth=0.5)
        ax.legend(
            title=(
                f"Star ID: {ID}\n"
                f"Teff={param[0]:.0f}K  logg={param[1]:.2f}  Z={param[2]:.2f}\n"
                f"kernel={best_settings['kernel']}\n"
                f"epsilon={best_settings['epsilon']:.2f}  smoothing={best_settings['smoothing']:.1e}\n"
                f"MSE={best_settings['mse']:.2e}\n"
                f"n_neigh={best_settings['n_neighbours']}"
            ),
            fontsize       = 6,
            title_fontsize = 8
        )
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        print(f"Saved: {txt_path}, {png_path}")

        return {
            'ID'    : ID,
            'result': best_settings
        }
    except Exception as e:
        print(f"Failed for star {ID}: {e}")
        return None


def interpall(max_passes=100):
    """Main function to interpolate all stellar spectra."""
    IDs, Teffs, loggs, Zs = param_retrieve()
    param_vectors = np.vstack((Teffs, loggs, Zs)).T

    # Set up data and parameters
    spectra     = []
    good_params = []
    good_IDs    = []
    for ID, p in zip(IDs, param_vectors):
        try:
            spec = get_spectra(ID)
            spectra.append(spec[:, 1])
            good_params.append(p)
            good_IDs.append(ID)
        except Exception as e:
            print(f"Failed to load {ID}: {e}")

    if len(spectra) == 0:
        raise ValueError("No suitable data found for interpolation.")

    os.makedirs('./Stellar_Spectra', exist_ok=True)

    good_params        = np.array(good_params)
    spectra            = np.array(spectra)
    wavelength         = get_spectra(good_IDs[0])[:, 0]
    scaler             = StandardScaler()
    good_params_scaled = scaler.fit_transform(good_params)

    # Precompute Nearest Neighbours
    print("Precomputing nearest neighbours...")
    nbrs = NearestNeighbors(n_neighbors=max_neigh, algorithm='auto').fit(good_params_scaled)
    distances_all, indices_all = nbrs.kneighbors(good_params_scaled)

    indices_list   = []
    distances_list = []
    for i in range(len(good_IDs)):
        indices   = indices_all[i]
        distances = distances_all[i]

        # Remove self-match if exists
        mask      = indices != i
        indices   = indices[mask]
        distances = distances[mask]

        indices_list.append(indices)
        distances_list.append(distances)

    print("Nearest neighbours precomputed.")

    # Interpolate each star's spectrum in parallel for speed - kachaow
    results_dict = {}
    all_errors   = []
    futures      = []

    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    with ProcessPoolExecutor(max_workers=2) as executor:
        for i, (param, ID, true_spectra) in enumerate(zip(good_params, good_IDs, spectra)):
            futures.append(
                executor.submit(interpolate_one_star, i, param, ID, true_spectra,
                                indices_list, distances_list, good_IDs, spectra, scaler, wavelength,
                                epsilons, smoothings, kernels, min_neigh, good_params_scaled)
            )

        for future in as_completed(futures):
            result = future.result()
            if result:
                ID = result['ID']
                results_dict[ID] = result['result']
                all_errors.append(result['result']['mse'])

    with open('./Stellar_Spectra/interpolation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("Saved interpolation results: ./Stellar_Spectra/interpolation_results.json")

    # Convergence plot
    all_errors = np.array(all_errors)
    plt.figure(figsize=(10, 5))
    plt.plot(all_errors, 'o-', markersize=3, linewidth=0.8, label='MSE per spectrum')
    plt.axhline(np.mean(all_errors), color='r', linestyle='--', label='Mean MSE')
    plt.xlabel('Star Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Convergence Summary: Interpolated vs Original Spectra')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Stellar_Spectra/convergence_summary.png')
    plt.close()

    print("Saved convergence summary: ./Stellar_Spectra/convergence_summary.png")
