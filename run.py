# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:11:56 2019

@author: chrisw
"""

import stpars
import SSP_model
import interp
import numpy as np
from astropy.io import ascii

import FitModelAndCheckSuitability
import SSP_comparison

import os
import shutil
from datetime import datetime


def CopyFiles():
    """
    Copy files to a backup directory with a timestamp.
    """
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    backup_root = f'backup_{timestamp}'

    # List of folders to back up
    folders_to_copy = ['Stellar_pars', 'Stellar_Spectra', 'SuitabilityPlots']

    # Create the backup root directory
    os.makedirs(backup_root, exist_ok=True)

    # Copy each directory
    for folder in folders_to_copy:
        if os.path.exists(folder):
            dest = os.path.join(backup_root, folder)
            shutil.copytree(folder, dest)
            print(f'Copied {folder} -> {dest}')
        else:
            print(f'Skipped missing folder: {folder}')
    
    return


# Set the manual parameters for the stellar population synthesis
n_ms = 25
n_rg = 25
Z    = np.log10(0.03/0.012)
feh  = 0.281
afe  = 0.0
age  = 10
lmin = 9.353139996530000644e+03
lmax = 2.410741666080859795e+04
imf  = 'kroupa'
dl   = 0.1
iso  = 'padova' # options are padova or dartmouth


## Do an initial interpolation of all existing stars - grid search to find best parameters and save them
interp.interpall()

## Fit the model to the data, produce some plots to check the fit
FitModelAndCheckSuitability.main()

## Copy the files to a backup directory with a timestamp for now
CopyFiles()

## Generate stellar parameters for the chosen stellar constituency in isochrone space
stpars.stpars(n_ms, n_rg, feh, afe, age, fig = False, iso = iso)

## Use the best fit model parameters to produce a new set of stars for the chosen stellar parameters
interp.interpall(generated_stars=True, use_model=True, n_ms=n_ms, n_rg=n_rg, feh=feh, afe=afe, age=age, Z=Z,)
## Or refit the parameters to the new set of stars from scratch
# interp.interpall(generated_stars=True, n_ms=n_ms, n_rg=n_rg, feh=feh, afe=afe, age=age, Z=Z,)

## Generate an SSP model with the chosen stellar parameters and compare to existing
SSP_model.ssp_model(Z, fwhm = 2.5, feh = feh, afe = afe, age = age, imf = imf, n_ms = n_ms, n_rg = n_rg, dl = dl, iso = iso)

## Generate comparative plots
# SSP_comparison.RunAll() # Only works for this example TODO: Fix naming for all combinations (if valid)