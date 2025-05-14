import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a backend that does not require a display, for the multiprocessing cores
import matplotlib.pyplot as plt

import os
import json, jsonlines
import joblib

from scipy.interpolate      import RBFInterpolator
from scipy.spatial          import distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, roc_curve, auc
from sklearn.neighbors      import NearestNeighbors

from concurrent.futures import ProcessPoolExecutor, as_completed

from retrieve_irtf      import param_retrieve, get_spectra, set_spectra_name

import stpars
from astropy.io import ascii
import pandas as pd

debug = False

# Interpolation settings
epsilons   = np.linspace(0.5, 5.0, 10)
smoothings = np.logspace(-1, 0, 10)
kernels    = ['gaussian'] #, 'multiquadratic']
min_neigh  = 4
max_neigh  = 6

max_workers = os.cpu_count()
model_path  = os.path.join("Stellar_Spectra", "pretrained_interp_model.joblib")

def evaluate_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def load_best_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Pretrained model not found at: " + model_path)

    return None

def interpolate_one_star(i, param, ID, true_spectra, indices_list, distances_list, good_IDs, spectra, scaler, wavelength, epsilons, smoothings, kernels, min_neigh, good_params, 
                         generated_stars=False, best_model=None):
    
    if debug: print(f"Interpolating star {ID} with parameters {param}")

    try:
        target_param_scaled = scaler.transform([param])
        indices             = indices_list[i]
        distances           = distances_list[i]

        if len(indices) < min_neigh:
            print(f"Skipping {ID}: not enough neighbours")
            return None

        # Load a model for fitting
        if best_model:
            # Predict interpolation hyperparameters from stellar parameters
            input_df     = pd.DataFrame([param], columns=['Teff', 'logg', 'Z'])
            model_bundle = best_model

            models         = model_bundle['models']
            label_encoders = model_bundle.get('label_encoders', {})

            kernel        = models['kernel'].predict(input_df)[0]
            if 'kernel' in label_encoders:
                kernel = label_encoders['kernel'].inverse_transform([int(kernel)])[0]

            epsilon       = models['epsilon'].predict(input_df)[0]
            smoothing     = models['smoothing'].predict(input_df)[0]
            n_neigh       = int(np.round(models['n_neighbours'].predict(input_df)[0]))

            neighbour_params  = good_params[indices[:n_neigh]]
            neighbour_spectra = spectra[indices[:n_neigh]]
            model             = RBFInterpolator(neighbour_params, neighbour_spectra,
                                               kernel=kernel, epsilon=epsilon, smoothing=smoothing)
            int_spectra       = model(target_param_scaled)[0]

            # Skip ROC if true_spectra is zero-filled
            if np.any(true_spectra > 0.01):
                fpr, tpr, _ = roc_curve(true_spectra > 0.01, int_spectra)
                roc_auc     = auc(fpr, tpr)
                roc_data    = {
                    'ID': ID, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc
                }
                with jsonlines.open('./Stellar_Spectra/ROC.jsonl', 'a') as writer:
                    writer.write(roc_data)

            output   = np.column_stack((wavelength, int_spectra))
            filename = set_spectra_name(param[0], param[1], param[2])
            txt_path = './Stellar_Spectra/' + filename
            png_path = txt_path + '.png'
            np.savetxt(txt_path, output)
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(wavelength, int_spectra, 'b', label='Interpolated Spectrum', linewidth=0.5)
            ax.plot(wavelength, true_spectra, 'r--', label='True Spectrum', linewidth=0.5)
            ax.legend(fontsize=6)
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.ylabel('Flux')
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()

            print(f"Saved: {txt_path}, {png_path}")

            return {
                'ID': ID,
                'result': best_model
            }


        # Otherwise, perform a grid search for the best parameters

        best_mse      = float('inf')
        best_settings = {}
        for n_neigh in range(min_neigh, len(indices) + 1):
            neighbour_params  = good_params[indices[:n_neigh]]
            neighbour_spectra = spectra[indices[:n_neigh]]

            for epsilon in epsilons:
                for smoothing in smoothings:
                    for kernel in kernels:
                        if kernel in ["linear", "thin_plate_spline", "cubic", "quintic"] and epsilon != 1:
                            continue
                        try:
                            model = RBFInterpolator(neighbour_params, neighbour_spectra,
                                                    kernel=kernel, epsilon=epsilon, smoothing=smoothing)
                            int_spectra = model(target_param_scaled)[0]
                            mse         = evaluate_error(true_spectra, int_spectra)
                            if mse < best_mse:
                                best_mse = mse
                                best_settings = {
                                    'epsilon': epsilon,
                                    'smoothing': smoothing,
                                    'kernel': kernel,
                                    'mse': mse,
                                    'n_neighbours': n_neigh,
                                    'used_distances': distances[:n_neigh].tolist(),
                                    'used_IDs': [good_IDs[j] for j in indices[:n_neigh].tolist()]
                                }
                        except Exception:
                            continue

        if not best_settings:
            print(f"Skipping star {ID} due to interpolation failure")
            return None

        neighbour_params  = good_params[indices[:best_settings['n_neighbours']]]
        neighbour_spectra = spectra[indices[:best_settings['n_neighbours']]]
        model = RBFInterpolator(neighbour_params, neighbour_spectra,
                                kernel=best_settings['kernel'],
                                epsilon=best_settings['epsilon'],
                                smoothing=best_settings['smoothing'])
        int_spectra = model(target_param_scaled)[0]

        if np.any(true_spectra > 0.01):
            fpr, tpr, _ = roc_curve(true_spectra > 0.01, int_spectra)
            roc_auc     = auc(fpr, tpr)
            roc_data = {'ID': ID, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc}
            with jsonlines.open('./Stellar_Spectra/ROC.jsonl', 'a') as writer:
                writer.write(roc_data)

        output   = np.column_stack((wavelength, int_spectra))
        filename = set_spectra_name(param[0], param[1], param[2])
        txt_path = './Stellar_Spectra/' + filename
        png_path = txt_path + '.png'
        np.savetxt(txt_path, output)
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(wavelength, int_spectra, 'b', label='Interpolated Spectrum', linewidth=0.5)
        ax.plot(wavelength, true_spectra, 'r--', label='True Spectrum', linewidth=0.5)
        ax.legend(fontsize=6)
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        print(f"Saved: {txt_path}, {png_path}")

        return {'ID': ID, 'result': best_settings}

    except Exception as e:
        print(f"Failed for star {ID}: {e}")
        return None


def interpall(max_passes=100, generated_stars=False, n_ms=None, n_rg=None, feh=None, afe=None, age=None, Z=None, use_model=False, best_model=None):

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

    print("Precomputing nearest neighbours...")
    nbrs = NearestNeighbors(n_neighbors=max_neigh, algorithm='auto').fit(good_params_scaled)
    distances_all, indices_all = nbrs.kneighbors(good_params_scaled)
    print("Done precomputing.")

    settings_list = [(k, e, s) for k in kernels for e in epsilons for s in smoothings]

    results_dict = {}
    all_errors   = []
    futures      = []

    if generated_stars and (None in (n_ms, n_rg, feh, afe, age, Z)):
        raise ValueError("All arguments must be provided when generated_stars=True")

    if use_model:
        try:
            best_model = load_best_model()
            print("Loaded pre-trained model from disk.")
        except Exception as e:
            raise FileNotFoundError("Pre-trained model not found. Please train and save it using SuitabilityPlots.py.")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        if generated_stars:

            parsfile   = stpars.set_stpars_filename(n_ms, n_rg, feh, afe, age)
            t          = ascii.read(parsfile)
            new_params = np.array([[row[0], row[1], Z] for row in t])

            for i, param in enumerate(new_params):
                fake_id        = f"GEN_{i}"
                dummy_flux     = np.zeros(spectra.shape[1])
                distances      = cdist([scaler.transform([param])[0]], good_params_scaled)[0]
                sorted_idx     = np.argsort(distances)
                indices        = sorted_idx[:max_neigh]
                dists          = distances[indices]

                mask           = np.any(good_params[indices] != param, axis=1)
                indices        = indices[mask]
                dists          = dists[mask]

                futures.append(
                    executor.submit(interpolate_one_star, 0, param, fake_id, dummy_flux,
                                    [indices], [dists], good_IDs, spectra, scaler, wavelength,
                                    epsilons, smoothings, kernels, min_neigh, good_params_scaled, generated_stars,
                                    best_model=best_model)
                )

        else:
            indices_list   = []
            distances_list = []
            for i in range(len(good_IDs)):
                indices   = indices_all[i]
                distances = distances_all[i]

                mask      = indices != i
                indices   = indices[mask]
                distances = distances[mask]

                indices_list.append(indices)
                distances_list.append(distances)

            for i, (param, ID, true_spectra) in enumerate(zip(good_params, good_IDs, spectra)):
                futures.append(
                    executor.submit(interpolate_one_star, i, param, ID, true_spectra,
                                    indices_list, distances_list, good_IDs, spectra, scaler, wavelength,
                                    epsilons, smoothings, kernels, min_neigh, good_params_scaled, generated_stars,
                                    best_model=best_model)
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

    if not generated_stars:
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
