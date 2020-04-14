#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import CubicSpline as spline
import math
import os

wl = np.linspace(300, 700., 401)
interp_func = None

def fq_table_interp():
    ncfile = Dataset(os.environ['OCDATAROOT']+'/common/morel_fq.nc', 'r')
    data = ncfile['foq'][:,:,:,:,:]

    grdwvl = ncfile['wave'][:]
    grdchl = np.log10(ncfile['chl'][:])
    grdsolz = ncfile['solz'][:]
    grdviewz = ncfile['senz'][:]
    grdrelaz = ncfile['phi'][:]

    interp_func = rgi((grdwvl, grdsolz, grdchl, grdviewz, grdrelaz), data,\
                      fill_value=None,bounds_error=False)
    
    return interp_func

# fq = get_fq(w,s,chl_in,n,a,foq_data)
# purpose: estimate f/Q for a given suite of observations
# references:
# 1) Morel and Gentilli, Applied Optics, 35(24), 1996
# 2) Morel et al., Applied Optics, 41(3), 2002
# input:
# 1) wavelength float [w]
# 2) solar zenith angle float [s]
# 3) chl concentration float [chl_in]
# 4) nadir zenith angle float [n]
# 5) azimuth angle float [a]
# 6) f/Q table array fltarr [foq_data]
# output: f/Q correction term float [fq]
# 8.20.2003 - translated from get_foq.c by PJW, SSAI

def get_fq(solz, viewz, relaz, chl, wvl=None):
    global wl

    global interp_func
    
    if interp_func is None:
        interp_func = fq_table_interp()
        print("Initializing f/Q interpolation function")

    runspline=True
    if wvl is None:
        wvl = wl
        runspline = False

    interp_arr = np.empty((len(wvl),5))
    interp_arr[:,0] = wvl
    interp_arr[:,1] = solz
    interp_arr[:,2] = np.log10(chl)
    interp_arr[:,3] = viewz
    interp_arr[:,4] = relaz

    result = interp_func(interp_arr)

    if runspline:
        cs = spline(wvl, result)
        result = cs(wl)
         
    return result 


def morel_brdf(chl, solz, viewz, relaz, wvl=None, corr=False):
    
    h2o = 1.34
    thetap = math.degrees(math.asin(math.sin(math.radians(viewz)) / h2o))
    
    fqA = np.array(get_fq(solz, thetap, relaz, chl, wvl=wvl))
    fq0 = np.array(get_fq(0.0,  0.0,   0.0, chl, wvl=wvl))
    
    if corr:
        fq = fq0 / fqA 
    else:
        fq = fqA
        
    return fq

    
def main():
    """
    Primary driver for stand-alone version
    """
    __version__ = '1.0.0'

    parser = argparse.ArgumentParser(description=\
        'calculate Morel brdf correction')

    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    
    parser.add_argument('chl', type=float, default=None,help='input chlorophyll')
    parser.add_argument('solz', type=float, default=None,help='input solar zenith angle')
    parser.add_argument('viewz', type=float, default=None,help='input view zenith angle')
    parser.add_argument('relaz', type=float, default=None,help='input relative azimuth angle')
    parser.add_argument('--output_file',type=str, default=None, help="optional output filename; default: STDOUT")

    args = parser.parse_args()
    
    global wl
    wvl = [300.,350.,412.5, 442.5, 490., 510., 560., 620., 660.,700.]
    
    brdf = morel_brdf(args.chl,args.solz,args.viewz,args.relaz,wvl=wvl,corr=True)
    
    if args.output_file:
        ofile = open(args.output_file,'w')
        ofile.write("/begin_header\n")
        ofile.write("! Output of Model BRDF function\n")
        ofile.write("! chlorophyll: %f\n" % args.chl)
        ofile.write("! solar zenith angle: %f\n" % args.solz)
        ofile.write("! view zenith angle: %f\n" % args.viewz)
        ofile.write("! relative azinuth angle: %f\n" % args.relaz)
        ofile.write("/missing=-999\n")
        ofile.write("/delimiter=space\n")
        ofile.write("/fields=wavelength,brdf\n")
        ofile.write("/units=nm,dimensionless\n")
        for i in range(len(brdf)):
            ofile.write('%7.2f %12.9f\n' % (wl[i],brdf[i]))
        ofile.close()
    else:
        for i in range(len(brdf)):
            print('%7.2f %12.9f' % (wl[i],brdf[i]))

    
# The following allows the file to be imported without immediately executing.
if __name__ == '__main__':
    sys.exit(main())