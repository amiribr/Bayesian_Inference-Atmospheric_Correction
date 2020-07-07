#!/usr/bin/env python3
import numpy as np
import argparse
import sys
from netCDF4 import Dataset
# function get_f0.pro
# [structure of f0 and lambda] = get_f0(avg_wl=[array of wl])
# and avg_wl=[array of wl] calculated 10 nm average for each wl
global curr_direc
curr_direc = '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/src/LUT_generator' + '/orm_morel/'

def get_f0(wvls=None):

    ncfile = Dataset(curr_direc + 'thuillier2003_f0.nc', 'r')
    
    wl = ncfile['wavelength'][:]
    f0 = ncfile['irradiance'][:]
    
    wl_save = wl
    f0_save = f0
    
    if wvls:    
        avg_f0 = np.empty(len(wvls))
        avg_f0[:] = np.nan
    
        for i in range(len(wvls)):
            idx = np.where((wl >= wvls[i]-5.) & ( wl <= wvls[i]+5.))
            if idx:
                avg_f0[i] = np.mean(f0[idx])
    	
    
        f0_save = avg_f0
        wl_save = wvls
    
    
    result = (f0_save, wl_save )
    
    return result

def main():
    """
    Primary driver for stand-alone version
    """
    __version__ = '1.0.0'

    parser = argparse.ArgumentParser(description=\
        'Generates hyperspectral (1nm) or 10nm bandpass averaged \
        extraterrestrial irradiance spectrum using Thuillier (2002): \
        ')

    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    
    parser.add_argument('--wvl', type=float,nargs='+', default=None,help='wavelength(s) to generate')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    
    verbose = args.verbose

    if verbose and args.output_file:
        print("Generating extraterrestrial irradiance spectrum ...")
        
    
    print(get_f0(wvls=args.wvl))
    
# The following allows the file to be imported without immediately executing.
if __name__ == '__main__':
    sys.exit(main())
