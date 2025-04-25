import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from retrieve_irtf import param_retrieve, get_spectra, set_spectra_name
import os

def evaluate_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def interpall(max_passes=10):
    # Retrieve parameters and spectra
    IDs, Teffs, loggs, Zs = param_retrieve()
    param_vectors = np.vstack((Teffs, loggs, Zs)).T

    spectra = []
    good_params = []
    good_IDs = []
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

    good_params = np.array(good_params)
    spectra = np.array(spectra)
    wavelength = get_spectra(good_IDs[0])[:, 0]  # Assuming consistent wavelength

    # Grid search for best interpolation model
    best_error = float('inf')
    best_model = None
    best_params = {}
    epsilons = np.logspace(-1, 1, 5)
    smoothings = np.logspace(-3, -1, 5)

    passes = 0
    for epsilon in epsilons:
        for smoothing in smoothings:
            try:
                model = RBFInterpolator(good_params, spectra, kernel='gaussian',
                                        epsilon=epsilon, smoothing=smoothing)
                pred = model(good_params)
                error = evaluate_error(spectra, pred)
                if error < best_error:
                    best_error = error
                    best_model = model
                    best_params = {'epsilon': epsilon, 'smoothing': smoothing}
            except Exception as e:
                print(f"Interpolation failed for epsilon={epsilon}, smoothing={smoothing}: {e}")
            passes += 1
            if passes >= max_passes:
                break
        if passes >= max_passes:
            break

    if best_model is None:
        raise RuntimeError("No successful interpolation model could be built.")

    print(f"Best interpolation: epsilon={best_params['epsilon']}, smoothing={best_params['smoothing']}")

    # Interpolate and save results
    for p in good_params:
        int_spectra = best_model(np.array([p]))[0]
        output = np.column_stack((wavelength, int_spectra))
        filename = set_spectra_name(p[0], p[1], p[2])
        txt_path = './Stellar_Spectra/' + filename
        png_path = txt_path + '.png'

        # Save spectrum
        np.savetxt(txt_path, output)

        # Save plot
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(wavelength, int_spectra, 'b', label='Interpolated Spectra', linewidth=0.5)
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.savefig(png_path)
        plt.close()

        print(f"Saved: {txt_path}, {png_path}")

    # --- Convergence summary plot ---
    predicted_all = best_model(good_params)
    errors = np.mean((spectra - predicted_all)**2, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(errors, 'o-', markersize=3, linewidth=0.8, label='MSE per spectrum')
    plt.axhline(np.mean(errors), color='r', linestyle='--', label='Mean MSE')
    plt.xlabel('Star Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Convergence Summary: Interpolated vs Original Spectra')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Stellar_Spectra/convergence_summary.png')
    plt.close()

    print("Saved convergence summary: ./Stellar_Spectra/convergence_summary.png")
