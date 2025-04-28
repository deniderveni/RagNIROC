import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.interpolate import RBFInterpolator
from retrieve_irtf import param_retrieve, get_spectra, set_spectra_name
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def interpolate_one_star(i, param, ID, true_spectra, indices_list, distances_list, good_IDs, spectra, scaler, wavelength, epsilons, smoothings, kernels, min_neigh, good_params):
    try:
        target_param_scaled = scaler.transform([param])

        indices   = indices_list[i]
        distances = distances_list[i]

        if len(indices) < min_neigh:
            print(f"Skipping {ID}: not enough neighbours")
            return None

        best_mse      = float('inf')
        best_settings = {}

        for n_neigh in range(min_neigh, len(indices) + 1):
            train_params  = scaler.transform(np.array([good_params[j] for j in indices[:n_neigh]]))
            train_spectra = np.array([spectra[j] for j in indices[:n_neigh]])

            for epsilon in epsilons:
                for smoothing in smoothings:
                    for kernel in kernels:
                        try:
                            model = RBFInterpolator(train_params, train_spectra,
                                                    kernel=kernel,
                                                    epsilon=epsilon,
                                                    smoothing=smoothing)
                            int_spectra = model(target_param_scaled)[0]
                            mse = evaluate_error(true_spectra, int_spectra)

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

        # Refit with the best setup
        train_params  = scaler.transform(np.array([good_params[j] for j in indices[:best_settings['n_neighbours']]]))
        train_spectra = np.array([spectra[j] for j in indices[:best_settings['n_neighbours']]])
        model         = RBFInterpolator(train_params, train_spectra,
                                        kernel=best_settings['kernel'],
                                        epsilon=best_settings['epsilon'],
                                        smoothing=best_settings['smoothing'])
        int_spectra = model(target_param_scaled)[0]

        output   = np.column_stack((wavelength, int_spectra))
        filename = set_spectra_name(param[0], param[1], param[2])
        txt_path = './Stellar_Spectra/' + filename
        png_path = txt_path + '.png'

        np.savetxt(txt_path, output)

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
            fontsize=6,
            title_fontsize=8
        )
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        print(f"Saved: {txt_path}, {png_path}")

        return {
            'ID': ID,
            'result': best_settings
        }
    except Exception as e:
        print(f"Failed for star {ID}: {e}")
        return None

def interpall(max_passes=100):
    IDs, Teffs, loggs, Zs = param_retrieve()
    param_vectors = np.vstack((Teffs, loggs, Zs)).T

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

    epsilons   = np.logspace(-1, 1, 5)
    smoothings = np.logspace(-3, -1, 5)
    kernels    = ['gaussian', 'linear', 'cubic', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric']
    min_neigh  = 4
    max_neigh  = 10

    # --- Precompute Nearest Neighbours ---
    print("Precomputing nearest neighbours...")
    nbrs = NearestNeighbors(n_neighbors=max_neigh, algorithm='auto').fit(good_params_scaled)
    distances_all, indices_all = nbrs.kneighbors(good_params_scaled)

    indices_list   = []
    distances_list = []
    for i in range(len(good_IDs)):
        indices   = indices_all[i]
        distances = distances_all[i]

        # Remove self-match if exists
        mask = indices != i
        indices   = indices[mask]
        distances = distances[mask]

        indices_list.append(indices)
        distances_list.append(distances)

    print("Nearest neighbours precomputed.")

    results_dict = {}
    all_errors   = []
    futures      = []

    with ProcessPoolExecutor() as executor:
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

    # --- Convergence plot ---
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
