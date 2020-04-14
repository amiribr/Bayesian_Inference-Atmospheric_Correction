#!/usr/bin/env python3
import sys
sys.path.insert(
    0, '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/src/LUT_generator/orm_morel')

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import interp2d
import requests
from get_f0 import get_f0
from morel_read import morel_read
import collections
import argparse
import math
import sys
import os
global curr_direc
curr_direc = '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/src/LUT_generator' + '/orm_morel/'
__version__ = '1.0.0'
verbose = False

# Global variables
wl = np.linspace(300, 700., 401)
verbose = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COPIED FROM generate_gas_transmittance.py!
# ..consider making this a utility function
#
# TODO: use $OCDATAROOT/common/SensorInfo.json as starter dict
# read file

#sensorInfoFile = os.path.join(os.environ['OCDATAROOT'] , 'common','SensorInfo.json')
#with open(sensorInfoFile, 'r') as myfile:
#    sensorDefs=myfile.read()
#
## parse file
#sensors = json.loads(sensorDefs)
#print(sensors)

sensors = collections.defaultdict(dict)

sensors['modisa']['name'] = "MODIS-Aqua"
sensors['modisa']['rsr'] = "HMODISA_RSRs.txt"
sensors['modisa']['wvl'] = [412.,443.,469.,488.,531.,551.,555.,645.,667.,678.,748.,859.,869.,1240.,1640.,2130.]
sensors['modist']['name'] = "MODIS-Terra"
sensors['modist']['rsr'] = "HMODIST_RSRs.txt"
sensors['modisa']['wvl'] = [412.,443.,469.,488.,531.,551.,555.,645.,667.,678.,748.,859.,869.,1240.,1640.,2130.]
sensors['seawifs']['name'] = "SeaWiFS"
sensors['seawifs']['rsr'] = "SeaWiFS_RSRs.txt"
sensors['seawifs']['wvl'] = [412.,443.,490.,510.,555.,670.,765.,865.]
sensors['meris']['name'] = "MERIS"
sensors['meris']['rsr'] = "MERIS_RSRs.txt"
sensors['meris']['wvl'] = [413.,443.,490.,510.,560.,620.,665.,681.,709.,754.,762.,779.,865.,885.,900.]
sensors['octs']['name'] = "OCTS"
sensors['octs']['rsr'] = "OCTS_RSRs.txt"
sensors['octs']['wvl']=[412.,443.,490.,520.,565.,670.,765.,865.]
sensors['czcs']['name'] = "CZCS"
sensors['czcs']['rsr'] = "CZCS_RSRs.txt"
sensors['czcs']['wvl'] = [443.,520.,550.,670.]
sensors['viirsn']['name'] = "VIIRS-SNPP"
sensors['viirsn']['rsr'] = "VIIRSN_IDPSv3_RSRs.txt"
sensors['viirsn']['wvl'] = [410.,443.,486.,551.,671.,745.,862.,1238.,1601.,2257.]
sensors['viirsj1']['name'] = "VIIRS-JPSS1"
sensors['viirsj1']['rsr'] = "VIIRS1_RSRs.txt"
sensors['viirsn']['wvl'] = [411.,445.,489.,556.,667.,746.,868.,1238.,1604.,2258.]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_sensor_RSR (sensor):
    ''' Reads a sensor spectral response (RSR) from the OCW through https
        Input: Sensor name (string)
        Output: normalized Rsr (numpy 2-d array - wavelength x sensor band),
                wavelength,
                sensor band labels,
    '''
    
    defpath = 'https://oceancolor.gsfc.nasa.gov/docs/rsr/'
    webpath = 'undefined'
    
    try:
        webpath = defpath + sensors[sensor]['rsr']
        if verbose:
            print("Reading spectral response function from %s" % webpath)
    except:
        print("Unkown sensor")
        sys.exit()
        
    # Read URL text file that contains sensors RSR
    lines = requests.get(webpath)
    if (lines.status_code == requests.codes.ok):
        data = lines.text.split()

        c = -1
        Rsr = np.empty([10000,10000])
        for line in (data):
            c = c + 1
            if (line[0:7] == '/fields'):
                labels = line[8:].split(',')
            if (line == '/end_header'):
                c = c + 1
                break
        Rsr = np.reshape((data[c:]),[int(len(data[c:])/len(labels)),len(labels)])
        Rsr = np.array(Rsr,dtype='float64')
        Rsr = np.clip(Rsr,0,1e6)
        wavelength = Rsr[:,0]
        Rsr = Rsr[:,1:] / Rsr[:,1:].max(axis=0)
        return wavelength, Rsr, labels

    else:
        print("Received error: %s" % lines.status_code)
        print("while attempting to retrieve %s" % webpath)
        sys.exit()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convolve_rsr(sensor, spectrum):
    (rsr_wvl,rsr,rsr_label) = read_sensor_RSR(sensor)
    # Get bands for the sensor and limit to the range of the global wl set
    global wl
    global sensors
    bands = np.array(sensors[sensor]['wvl'])
    bidx = np.where((bands >= np.amin(wl)) & (bands <= np.amax(wl)))
    bands = bands[bidx]

    rsr_spectrum = np.empty(len(bands))
    idx = np.where((rsr_wvl >= np.amin(wl)) & (rsr_wvl <= np.amax(wl)))
    sidx = np.where((wl >= np.amin(rsr_wvl)) & (wl <= np.amax(rsr_wvl)))

    for i,_ in enumerate(bands):
            rsr_spectrum[i] = np.trapz(spectrum[sidx]*rsr[idx].transpose()[i,:]) / \
                              np.trapz(rsr[idx].transpose()[i,:])

    return (bands, rsr_spectrum)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# backscattering coefficient
# Loisel and Morel 1998, Ciotti et al. 1999, Morel and Maritorena 2001 
#

def get_mm01_bb(chl):
    global wl

	# backscattering coefficient for seawater
    bbw = 0.0038 * np.power(400./wl,4.32)

	# Loisel and Morel 1998 bbp function
    bp550 = 0.416 * np.power(chl,0.766)

	# Morel and Maritorena 2001 bbp with v from Ciotti et al. 1999
    v = 0.768 * np.log10(chl) - 1.
    bbp = (0.002 + 0.01 * (0.5 - 0.25 * np.log10(chl)) \
           * np.power(wl / 550.,v)) * bp550

    bb = bbp + bbw

    return bb

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# diffuse attenuation coefficient
# Morel and Maritorena 2001 
# Kd = Kw + X chl^e
# requires file mm01_kbio.txt, which has columns for wl, kw, e, and X
#

def get_mm01_kd(chl):
    global wl
    data = Dataset(curr_direc + 'mm01_kbio.nc', 'r')

    k0 = data['X'][:] * np.power(chl,data['e'][:]) + data['kw'][:]

    cs = spline(data['wavelength'][:], k0)
    kd = cs(wl)

    return kd


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# mean cosine for downward flux
# A. Morel, personal communication, Summer 2006
# requires external subroutine morel_read.pro and associated data files
#

def get_morel_mud(chl, solz):

	mud = morel_read(chl, solz, mud=True)

	return mud


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# f and Q
# Morel et al. 2002, appendix b
# for use with in-water data (senz = 0 and relaz = 0)
# requires morel_fq.txt
#

def get_morel_fq(chl, solz):
    
    global wl

    morel_fq_appb = Dataset(curr_direc + 'morel_fq_appb.nc', 'r')

    wlut = morel_fq_appb['wavelength'][:]
    clut = morel_fq_appb['chlorophyll'][:]

    z = 1. - np.cos(np.deg2rad(solz))

    q1 = morel_fq_appb['q0'][:] + morel_fq_appb['sq'][:] * z
    
    interp_func = interp2d(wlut, clut, q1, kind='linear',fill_value=None)

    q = interp_func(wl,chl)

    f = morel_read(chl, solz, fp=True)

    return (f,q)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# [rrs] = morel_rrs(chl, wvl=wvl, lwn=lwn)
#
# Rrs = trans * (f / Q) * {bb / (a + bb)}, where (a + bb) ~ mud * Kd
# based mostly on Morel and Maritorena 2001
#
# defaults to 380 to 700-nm, used wvl keyword for 11-nm averages
# use lwn keyword to derive lwn instead of rrs
#

def morel_rrs(chl, solz=30, wvl=None, lwn=False, mm01iter=True):
	# define wavelength range
	#
    global wl
    global verbose
	# default solar zenith angle is 30-degrees
	#
    #solz = 30.0

	# backscattering coefficient
	#
    bb = get_mm01_bb(chl)

	# diffuse attenuation coefficient
	#
    kd = get_mm01_kd(chl)

	# mean cosine for downward flux
	#
    mud = get_morel_mud(chl, solz)

	# f and Q
	#
    (f,q) = get_morel_fq(chl, solz)

	# Morel and Maritorena 2001, Morel et al. 2007 iteration
	#

    u2 = mud
    r0 = f * bb / (u2 * kd + bb)

    if mm01iter:
    # iterate 3 times
        for i in range(3):
            u2 = mud * (1. - r0) / (1. + mud * r0 / 0.4)
            atot = 0.962 * kd * mud * (1. - r0 / f)
            r0 = f * bb / (atot + bb)
    else:
    # iterate until convergence
        last_r0 = r0 + 2.
        iterations=0
        close_enough = False
        while close_enough is False:
            iterations += 1
            u2 = mud * (1. - r0) / (1. + mud * r0 / 0.4)
            atot = 0.962 * kd * mud * (1. - r0 / f)
            r0 = f * bb / (atot + bb)
            close_enough = True
            for i in range(len(r0)):
                if not math.isclose(r0[i], last_r0[i],rel_tol = 1e-07):
                    last_r0[i] = r0[i]
                    close_enough = False
                   # print(i,iterations)
                    continue
            if close_enough:
                break

        if verbose:
            print("MM01 iterations: %d" % iterations)

    # Gordon et al. 1988 transmission across the air-sea interface
	#
    rho = 0.021
    rho_bar = 0.043
    r_eu = 0.48
    m = 1.34

    t = ((1. - rho) * (1. - rho_bar)) / (np.power(m,2) * (1. - r_eu * r0))
    rrs = t * r0 / q

	# lwn instead of rrs
	#
    if lwn:
        (f0,f0wl) = get_f0()
        widx = np.where((f0wl >= np.amin(wl)) & (f0wl <=np.amax(wl)))
        rrs = rrs * f0[widx]

	# 11-nm averages centered on discrete lambda
	#
    if wvl:
        dsc = np.empty(len(wvl))
        dsc[:] = np.nan
        
        for i in range(len(wvl)):
            v = np.where((wl >= wvl[i]-5.) & (wl <= wvl[i]+5.))
            if v:
                dsc[i] = np.mean(rrs[v])

        rrs = dsc
    
    return rrs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    Primary driver for stand-alone version
    """
    global verbose
    global wl
    global sensors

    parser = argparse.ArgumentParser(description=\
        'Generates hyperspectral (1nm) or sensor-specific Rrs (or Lwn) \
        spectra using the ocean reflectance model described in: \
        P. Jeremy Werdell, Sean W. Bailey, Bryan A. Franz, André Morel, and \
        Charles R. McClain, "On-orbit vicarious calibration of ocean color \
        sensors using an ocean surface reflectance model," \
        Appl. Opt. 46, 5649-5666 (2007), https://doi.org/10.1364/AO.46.005649')

    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('--sensor', type=str,
                        choices=['modisa',
                                 'modist',
                                 'viirsn',
                                 'viirsj1',
                                 'seawifs',
                                 'meris',
                                 'octs',
                                 'czcs'],
                        default=None, help='Sensor to mimic')
    parser.add_argument('--output_file', type=str, 
                        help='output netCDF LUT filename; default is STDOUT')
    parser.add_argument('chl', type=float,default=0.05,help='Chlorophyll concentration')
    parser.add_argument('--solz', type=float, default=30.,help='Solar zenith angle')
    parser.add_argument('--wvl', type=float,nargs='+', default=None,help='Wavelength(s) to generate')
    parser.add_argument('--lwn', action='store_true', help="ouput normalized Lw instead of Rrs")
    parser.add_argument('--disable_iter_limit', action='store_false', help="allow MM01 iteration to run to convergence")
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    verbose = args.verbose
    if args.sensor and args.wvl:
        print("Can't do both sensor and an input wavelength set")
        sys.exit(1)

    wvl = wl
    if args.wvl:
        wvl = args.wvl
        if verbose:
            print("Generating 11nm bandpass spectrum for in put wavelengths")

    if verbose and args.output_file:
        print("Generating Rrs spectrum %s ..." % args.output_file)

    otype = 'rrs'
    units = '1/sr'
    if args.lwn:
        otype = 'nLw'
        units = 'uW/cm^2/nm'

    spectrum = morel_rrs(args.chl, solz=args.solz, wvl=args.wvl, lwn=args.lwn, \
                         mm01iter=args.disable_iter_limit)

    # Convolve with sensor RSR, if desired...
    if args.sensor:
        if verbose:
            print("Generating spectrum based on the spectral response functions for %s" % args.sensor)
        (wvl, spectrum) = convolve_rsr(args.sensor,spectrum)

    if args.output_file:
        ofile = open(args.output_file,'w')
        ofile.write("/begin_header\n")
        ofile.write("! Output of Ocean Reflectance Model\n")
        ofile.write("! chlorophyll: %f\n" % args.chl)
        ofile.write("! solar zenith angle: %f\n" % args.solz)
        if args.sensor:
            ofile.write("! convolved with the spectral response function for %s\n" % sensors[args.sensor]['name'])
        ofile.write("/missing=-999\n")
        ofile.write("/delimiter=space\n")
        ofile.write("/fields=wavelength,%s\n" % otype)
        ofile.write("/units=nm,%s\n" % units)
        for i in range(len(spectrum)):
            ofile.write('%7.2f %12.9f\n' % (wvl[i],spectrum[i]))
        ofile.close()
    else:
        for i in range(len(spectrum)):
            print('%7.2f %12.9f' % (wvl[i],spectrum[i]))
# The following allows the file to be imported without immediately executing.
if __name__ == '__main__':
    sys.exit(main())
