import numpy as np
import xarray as xr
import scipy.interpolate as interp
from numba import njit

global koz
global tco2
global kno2
global fTh2o

koz = np.array([1.99E-03, 3.19E-03, 8.75E-03, 2.03E-02, 6.84E-02, 8.62E-02,
                9.55E-02, 7.38E-02, 4.89E-02, 3.79E-02, 1.24E-02, 2.35E-03,
                1.94E-03, 0.00E+00, 0.00E+00, 0.00E+00])

tco2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 0.99941, 0.98896, 0.96965])

kno2 = np.array([5.81E-19, 4.99E-19, 3.94E-19, 2.88E-19, 1.53E-19, 1.19E-19,
                 9.45E-20, 1.38E-20, 7.07E-21, 8.30E-21, 2.16E-21, 6.21E-23,
                 7.87E-23, 0.00E+00, 0.00E+00, 0.00E+00])

wvTbl = xr.open_dataset(
    '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/data/tables/MODISA_wv_txtbl.nc')

# interpolate the water vapor table
fTh2o = interp.interp1d(wvTbl.water_vapor, wvTbl.sensor_trans[4, :, :])


@njit
def ozone(O3, senz, solz):
    tau_oz = O3 * koz
    To3_sol = np.exp(-tau_oz / np.cos(np.deg2rad(solz)))
    To3_sen = np.exp(-tau_oz / np.cos(np.deg2rad(senz)))
    return To3_sol, To3_sen


@njit
def co2(senz, solz):
    Tco2_sol = tco2 * np.power(tco2, 1 / np.cos(np.deg2rad(solz)))
    Tco2_sen = tco2 * np.power(tco2, 1 / np.cos(np.deg2rad(senz)))
    return Tco2_sol, Tco2_sen


def h2o(cwv, senz, solz):
    wv_sol = cwv / np.cos(np.deg2rad(solz))
    wv_sen = cwv / np.cos(np.deg2rad(senz))
    Th2o_sol = fTh2o(wv_sol)
    Th2o_sen = fTh2o(wv_sen)
    return Th2o_sol, Th2o_sen


wvTbl.close()
