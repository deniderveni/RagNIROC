# RagNIROC: A new-age, ML interpolator for the Near Infrared spectrum -- based on [MIOLNIR](https://github.com/deniderveni/MIOLNIR)

MIOLNIR: Modelling and Interpolation across Observed Luminosities in the Near Infra Red
MIOLNIR utilised an interpolation method that has stringent limitations on its choice of stars. In other words, they must be worthy.
It was a custom, simplistic Radial Basis Function (RBF) interpolator - it was also my first ever full Python project! Built alongside Christopher Wilson (see MIOLNIR)

Meanwhile, RagNIROC uses a formal RBF interpolator to optimise fitting hyperparameters for nearby stellar spectra, with greater elegance and modularity.
It also begins to use regression techniques to create spectral models which, once I have greater confidence in its ability, will be saved and used in its finality to predict stellar spectra.


This repository is a WIP. It is fully functional but currently contains relics from MIOLNIR.
Over time, I will develop this into a formal, stand-alone package. For now, it serves as a proof-of-concept for the methodology.

## Usage

Run `run.py` to:
- Train and optimise a model synthetic spectra for existing stars
- Save the model for Stellar spectral synthesis
- Generate purely synthetic spectra for theoretical stars
- Create a Stellar Population based on trained model


`run.py` sets initial values for all the variables where necessary. It then runs the following programs:
  - **Old code** `stpars()` - This code calculates the Teff, logg and Z components of several stars along an isochrone. Currently, the only isochrone in use is Padova age 10Gyr, but outputs exist for all types.
  - `interpall()` - This creates interpolated spectra for all the stars contained in the stpars output files, using an RBF Interpolation scheme. It can also be used with an existing saved model to generate synthetic spectra for given parameters
  - `FitModelAndCheckSuitability.main()` - This read through the fit data, creates an optimised model for the breadth of chosen parameters, and saves this to disk. Also produces figures and data for assessment.
  - **Old code** `SSP_model()` - This combines all the spectra together to create an SSP.

There is also `SSP_comparison.py`:
  - This can be run to assess the suitability of the generated model in creating a complete SSP. **N.B: This is currently only intended for use for a single stellar make-up for showcasing**
  

## Installation

To use, Pyphot must first be installed:

```unix
pip install git+https://github.com/mfouesneau/pyphot
```

Any troubles see http://mfouesneau.github.io/docs/pyphot/.

Next just clone this repository
```unix
git clone https://github.com/deniderveni/RagNIROC
```

The contained codes and their functions:

## stpars.py

stpars.py contains 3 different functions: stpars(), gettracks() and set_stpars_filename().

**stpars(n_ms, n_rg, feh, afe, age, logg_cn = 3, fig = False, iso = 'DARTMOUTH')**
  - stpars uses the following required inputs:
    - n_ms = number of main sequence stars
    - n_rg = number of red giant stars
    - feh = iron abundance [Fe/H]
    - afe = [alpha/Fe]
    - age = age of the population (in Gyrs)
    
  - stpars uses the following additional inputs:
    - logg_cn = stars with logg <= logg_cn
    - fig = logical, whether you want the isochrone plotted or not
    - iso = isochrone model used, currently supports DARTMOUTH or PADOVA

stpars finds the appropriate isochrone for the desired inputs and calculates the Teff and logg of the n_ms stars on the main sequence and n_rg on the red giant branch. It outputs a table of the Teff's, logg's, masses, Hband logL's and their respective phases in a .dat file found in the folder Stellar_pars.

**gettracks(feh, afe, age, iso = 'DARTMOUTH')**

gettracks uses the previously defined parameters to find the correct isochrone. It outputs the file path of the isochrone selected.

**set_stpars_filename(n_ms, n_rg, feh, afe, age, logg_cn = 3)**

This outputs the filename and path for the parameter file given the previously defined inputs.

# Stellar Spectra Interpolation

This project generates interpolated stellar spectra using the IRTF spectral library and stellar parameters (Teff, logg, Z). It includes utilities for population synthesis, performance testing, and classification evaluation.

---

## interp.py

This script handles stellar spectra interpolation based on target stellar parameters.

### Functions

#### `interpolate(Teff_new, logg_new, Z_new, plot=True, save=True)`

Interpolates a spectrum from the IRTF library for a star with the given parameters.

**Parameters:**
- `Teff_new`: Effective temperature in Kelvin  
- `logg_new`: Surface gravity (log(g))  
- `Z_new`: Metallicity  
- `plot`: (bool) Save diagnostic plot to `Stellar_Spectra/` (default: `True`)  
- `save`: (bool) Save `.npy` file to `Stellar_Spectra/` (default: `True`)  

**Returns:**
- Wavelength array  
- Flux array (interpolated spectrum)  
- List of IDs of the three closest spectra used in interpolation

---

#### `interpall(generated_stars, use_model, n_ms, n_rg, feh, afe, age, Z)`

Interpolates spectra for an entire isochrone population using stellar parameters generated by `stpars.py`.

**Parameters: All optional**
- `generated_stars`: Bool, dictates whether synthetic stars are being created or if existing data is being created (requires all other arguments)
- `use_model`: Bool, dictates whether an existing saved model is used (if none exists, returns fault)
- `n_ms`: int, Number of main sequence stars  
- `n_rg`: int, Number of red giant stars  
- `feh`: float, [Fe/H] metallicity  
- `afe`: float, [α/Fe] abundance  
- `age`: float, Stellar population age (Gyr)  
- `Z`: float, Metallicity for interpolation

Saves spectra and plots in the `Stellar_Spectra/` directory.

## retrieve_irtf.py

Utilities for retrieving and processing IRTF spectral data and metadata.

### Functions

#### `param_retrieve()`

Reads `irtf_param.txt` and returns structured stellar parameter data.

**Returns:**
- Structured NumPy array with fields like ID, Teff, logg, Z, etc.

---

#### `get_spectra(ID)`

Retrieves a spectrum from the IRTF FITS files.

**Parameters:**
- `ID`: Spectrum ID string (e.g. `'IRL012'`)  

**Returns:**
- Wavelength array  
- Flux array

---

#### `set_spectra_name(Teff, logg, Z)`

Generates a standardised filename based on stellar parameters.

**Parameters:**
- `Teff`: Effective temperature  
- `logg`: Surface gravity  
- `Z`: Metallicity  

**Returns:**
- Formatted filename string for saving interpolated spectra

---

## SSP_model.py

SSP_model is an apdated version of https://github.com/marinatrevisan/SynSSP_PFANTnew.


# Data Outputs:

Several functions in this code produce some sort of figures or data dotted around in folders. Frankly, this was just poor planning and working with the ori
ginal MIOLNIR's structure without disrupting it too much, but this will be fixed sooner or later.

If you would like to see what the outputs look like without running this code, `git checkout main_w_output_data`, which is a branch that mirrors `main` but also uploads the output data, copied to a specific `test_outputs` folder.

In any case, here is a run down of what to expect:

- `Stellar_pars`:
    - In this example, this will only produce 1 file, which contains a datasets that has been generated from a theoretical isochrone model, being pulled from the set of given spectral data in `run.py`

- `Stellar_Spectra`:
    - This contains all of the data for individual synthesised (predicted) spectra:
        - `intspectra_Teff....`: Just csv files with synthesised data, used for the SSP population later
        - `intspectra_ ... .png`: The specific spectra plotted for visualisation. If you were replicating real spectra (i.e. `interpall(generated_stars=False)`), the plots will contain both real and synthetic data for visual comparison. Otherwise, only the synthesised data will be present
        - `convergence_summary.png`: A figure showing the RMS of each fitted star against the mean (only if `generated_stars=False`)
        - `interpolation_results.json`: A .json file with the interpolation information stored for training/testing later
        - `pretrained_interp_model.joblib`: The final predictive model after fitting against all data. This can be loaded later for a fast prediction instead of re-running the interpolation gridsearch (`interp.py`).

- `SuitabilityPlots`:
    - Contains Several sets of figures for general tracking of the fit performance:
        - `All / Above_0.1 / Below_0.1`: Contains histograms and scatter plots for the distributions of the final best fit parameters for the chosen parameters being iterated through.
            - `Above_0.1` and `Below_0.1` separates the data by those that were above and below 0.1 MSE, while `All` is everything. In other words, the data separated by the line in `convergence_summary.png`

    - `GoodFits`:
        - Scatter plots for the best parameters across all distributions and stars. Useful for quickly inspecting if there's some sort of patterns emerging (useless if fitting with too many or too few options).

    - `ML_Parameter_Predictions`: After running `FitModelAndCheckSuitability.main()`, The produced model will predict the parameters for each star. This folder contains a series of scatter plots with the predictions and visual aid for the final decision tree produced by the model

    - `Mean`/`All_ROC_Curves.png`: ROC curves for the true/false positive rate for the final prediction of real spectral data for mean of all stars / each star.
    
    -   predicted_best_parameters.csv`: A csv file for all of the best parameters

- `SSP_Spectra`:
    - A folder cotnaining the final SSP Spectra, including an illustrative figure with literature models, a csv of the data and an informative log
    - `ComparisonPlots`: A folder with more comprehensive comparative plots of the new RagNIRoC model vs the old MIOLNIR interpolation vs accepted models from literature (BaSS/MarS/GiRS)



