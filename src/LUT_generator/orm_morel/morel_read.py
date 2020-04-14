import argparse
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator as rgi
import os
global curr_direc
curr_direc = '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/src/LUT_generator' + '/orm_morel/'

interp_func_mud = None
interp_func_fn = None
interp_func_fp = None

def morel_table_interp(mud=False, fn=False, fp=False):
    ncfile = Dataset(curr_direc + 'morel_mud_f_fp.nc', 'r')
    data = None
    if mud: 
        data = ncfile['mud'][:,:,:]
    elif fn:
        data = ncfile['f'][:,:,:]
    elif fp:
        data = ncfile['fp'][:,:,:]

    if data is None:
    	print, ''
    	print, 'Error: please indicate mud, f, or fp'
    	print, ''
    	sys.exit(-1)

    grdwvl = ncfile['wavelength'][:]
    grdchl = ncfile['chlorophyll'][:]
    grdsolz = ncfile['solz'][:]

    interp_func = rgi((grdsolz, grdwvl, grdchl), data,fill_value=None,bounds_error=False)

    return interp_func


def morel_read(chl, solz, mud=False, fn=False, fp=False):

    global interp_func_mud
    global interp_func_fn
    global interp_func_fp

    if mud:
        if interp_func_mud is None:
            interp_func_mud = morel_table_interp(mud=True)
            print("Initializing mean cosine of downwelling irradiance interpolation function")
        interp_func = interp_func_mud
    elif fn:
        if interp_func_fn is None:
            interp_func_fn = morel_table_interp(fn=True)
            print("Initializing f interpolation function")
        interp_func = interp_func_fn
    elif fp:
        if interp_func_fp is None:
            interp_func_fp = morel_table_interp(fp=True)
            print("Initializing f-prime interpolation function")
        interp_func = interp_func_fp
    else:
        print, ''
        print, 'Error: please indicate mud, f, or fp'
        print, ''
        sys.exit(-1)

    wl = np.linspace(300, 700., 401)
    interp_arr = np.empty((len(wl),3))
    interp_arr[:,0] = solz
    interp_arr[:,1] = wl
    interp_arr[:,2] = chl

    return interp_func(interp_arr)

    
def main():
    """
    Primary driver for stand-alone version
    """
    __version__ = '1.0.0'

    parser = argparse.ArgumentParser(description=\
        'Read Morel tables')

    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    
    parser.add_argument('chl', type=float, default=None,help='input chlorophyll')
    parser.add_argument('solz', type=float, default=None,help='input solar zenith angle')

    args = parser.parse_args()
    
    
    print(morel_read(args.chl,args.solz,mud=True))
    
# The following allows the file to be imported without immediately executing.
if __name__ == '__main__':
    sys.exit(main())
