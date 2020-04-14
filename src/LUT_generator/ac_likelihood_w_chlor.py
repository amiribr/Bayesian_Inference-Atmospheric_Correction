import sys
sys.path.insert(
    0, '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/src/LUT_generator/')
from orm_morel.orm import convolve_rsr
from orm_morel.get_brdf import morel_brdf
from orm_morel.orm import morel_rrs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import xarray as xr
import scipy.optimize as opt
from tqdm import tqdm
import warnings
from numba import jit, njit, prange
import scipy.interpolate as interp
from pygetglint import getglint_iqu
import gastrans
import scipy.interpolate as interp

global wl
wl = np.array([412.,  443.,  469.,  488.,  531.,  547.,  555.,  645.,  667.,
               678.,  748.,  859.,  869., 1240., 1640., 2130.])


@njit
def fast_interp(grid, vals, points):
    out = np.empty((vals.shape[0]))
    #P = np.empty((1, 8))
    #Q = np.empty((1, 8))
    out = TriLinearInterp(grid, vals, points)
    return out


@njit
def mdot(A, B):
    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m, p))
    for i in range(0, m):
        for j in range(0, p):
            for k in range(0, n):
                C[i, j] += A[i, k] * B[k, j]
    return C


B = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 1, 0, 0, 0],
              [-1, 0, 1, 0, 0, 0, 0, 0], [-1, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, -1, 0, -1, 0, 1, 0], [1, -1, -1, 1, 0, 0, 0, 0],
              [1, -1, 0, 0, -1, 1, 0, 0], [-1, 1, 1, -1, 1, -1, -1, 1]])


@njit(parallel=True, fastmath=True)
def TriLinearInterp(grid, values, points):
    P = np.empty((values.shape[0], 8, 1))
    Q = np.empty((1, 8))
    p = np.empty((values.shape[0]))
    # x-axis
    x_idx0 = np.searchsorted(grid[0], points[0])
    x_idx1 = x_idx0 - 1
    dx = (points[0] - grid[0][x_idx0]) / (grid[0][x_idx1] - grid[0][x_idx0])
    # y-axis
    y_idx0 = np.searchsorted(grid[1], points[1])
    y_idx1 = y_idx0 - 1
    dy = (points[1] - grid[1][y_idx0]) / (grid[1][y_idx1] - grid[1][y_idx0])
    # z-axis
    z_idx0 = np.searchsorted(grid[2], points[2])
    z_idx1 = z_idx0 - 1
    dz = (points[2] - grid[2][z_idx0]) / (grid[2][z_idx1] - grid[2][z_idx0])

    P[:, 0, 0] = values[:, x_idx0, y_idx0, z_idx0]
    P[:, 1, 0] = values[:, x_idx0, y_idx0, z_idx1]
    P[:, 2, 0] = values[:, x_idx0, y_idx1, z_idx0]
    P[:, 3, 0] = values[:, x_idx0, y_idx1, z_idx1]
    P[:, 4, 0] = values[:, x_idx1, y_idx0, z_idx0]
    P[:, 5, 0] = values[:, x_idx1, y_idx0, z_idx1]
    P[:, 6, 0] = values[:, x_idx1, y_idx1, z_idx0]
    P[:, 7, 0] = values[:, x_idx1, y_idx1, z_idx1]

    Q[0, 0] = 1
    Q[0, 1] = dx
    Q[0, 2] = dy
    Q[0, 3] = dz
    Q[0, 4] = dx * dy
    Q[0, 5] = dy * dz
    Q[0, 6] = dz * dx
    Q[0, 7] = dx * dy * dz
    for i in prange(values.shape[0]):
        C = mdot(B, P[i])
        p[i] = np.multiply(Q, C[:, 0]).sum()
    return p


reflut = None
reflut_int = None
ρ = None

global taur_p0
global F0

F0 = np.array([172.912,	187.622, 205.878, 194.933, 185.747,
    186.539, 183.869, 157.811, 152.255, 148.052, 128.065, 97.174,
    95.824, 45.467, 23.977, 9.885])
reflut_fname = '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/data/tables/aerosol_ref_transmittance_LUT.nc'
print("Opening %s" % (reflut_fname))
reflut = xr.open_dataset(reflut_fname)

# Read in the rayleigh tables
ray_fname = '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/data/tables/Rayleigh_LUT.nc'
print("Opening %s" % (ray_fname))
raylut = xr.open_dataset(ray_fname)
taur_p0 = raylut.taur.values
global ray_grid
global rayrefI
global rayrefQ
global rayrefU
global order
ray_grid = (raylut.sigma_wind.values, raylut.solz.values, raylut.senz.values)
rayrefI = np.moveaxis(raylut.I.values.reshape((8, 45, 41, 3 * 16)), 3, 0)
rayrefQ = np.moveaxis(raylut.Q.values.reshape((8, 45, 41, 3 * 16)), 3, 0)
rayrefU = np.moveaxis(raylut.U.values.reshape((8, 45, 41, 3 * 16)), 3, 0)
order = raylut.norder.values

p0 = 1013.25

grid = (reflut.solz.values, reflut.phi.values, reflut.senz.values)
vals = reflut.ρ.values.reshape((80 * 5 * 16, 33, 19, 35))
tsol_vals = np.moveaxis(reflut.tsol.values.reshape(8, 10, 5, 16 * 33), 3, 0)
tsen_vals = np.moveaxis(reflut.tsen.values.reshape(8, 10, 5, 16 * 33), 3, 0)


@njit
def RayPressWang(airmass, pr):
    p0 = 1013.25
    x = ((-(0.6543 - 1.608 * taur_p0) + (0.8192 - 1.2541 * taur_p0)
          * np.log(airmass)) * taur_p0 * airmass)
    fac = (1.0 - np.exp(-x * pr / p0)) / (1.0 - np.exp(-x))
    return fac


@njit
def get_rayleigh(ray_points, phi, pr):
    I_interp = np.empty((3 * 16))
    Q_interp = np.empty((3 * 16))
    U_interp = np.empty((3 * 16))
    ρir = np.empty((1, 16))
    ρqr = np.empty((1, 16))
    ρur = np.empty((1, 16))
    phia_cos = np.empty((3, 1))
    phia_sin = np.empty((3, 1))
    I_interpm = np.empty((3, 16))
    Q_interpm = np.empty((3, 16))
    U_interpm = np.empty((3, 16))

    I_interp = fast_interp(ray_grid, rayrefI, ray_points)
    Q_interp = fast_interp(ray_grid, rayrefQ, ray_points)
    U_interp = fast_interp(ray_grid, rayrefU, ray_points)

    airmass = 1. / \
        np.cos(np.deg2rad(ray_points[1])) + 1. / np.cos(np.deg2rad(ray_points[2]))
    fac = RayPressWang(airmass, pr)

    I_interpm[0, :] = I_interp[0:16]
    I_interpm[1, :] = I_interp[16:32]
    I_interpm[2, :] = I_interp[32:]

    Q_interpm[0, :] = Q_interp[0:16]
    Q_interpm[1, :] = Q_interp[16:32]
    Q_interpm[2, :] = Q_interp[32:]

    U_interpm[0, :] = U_interp[0:16]
    U_interpm[1, :] = U_interp[16:32]
    U_interpm[2, :] = U_interp[32:]

    phia_cos[:, 0] = (np.cos(np.deg2rad(order * phi)) *
                      np.pi / np.cos(np.deg2rad(ray_points[1])))
    phia_sin[:, 0] = (np.cos(np.deg2rad(order * phi)) *
                      np.pi / np.sin(np.deg2rad(ray_points[1])))

    ρir = fac * mdot(I_interpm.T, phia_cos).T
    ρqr = fac * mdot(Q_interpm.T, phia_cos).T
    ρur = fac * mdot(U_interpm.T, phia_sin).T
    # ρir = fac*I_interpm.T.dot(np.cos(np.deg2rad(order*phi)))*np.pi/np.cos(np.deg2rad(ray_points[1]))
    # ρqr = fac*I_interpm.T.dot(np.cos(np.deg2rad(order*phi)))*np.pi/np.cos(np.deg2rad(ray_points[1]))
    # ρur = fac*I_interpm.T.dot(np.cos(np.deg2rad(order*phi)))*np.pi/np.sin(np.deg2rad(ray_points[1]))

    return ρir, ρqr, ρur


def aer_ref(reflut_int, rh, fmf_ret, τ_ret):
    ρa = interp.interpn(
        (reflut.humidity.values,
         reflut.fmf.values,
         reflut.tau.values),
        reflut_int,
        (rh,
         fmf_ret,
         τ_ret),
        bounds_error=False)
    return ρa

global tbl_grid
global dtheta
tbl_grid = (reflut.humidity.values, reflut.fmf.values, reflut.tau.values)
dtheta = reflut.dt_theta.values
#@njit
def aer_trans(rh, fmf_ret, τ_ret, solz, senz):
    tsol_interp = fast_interp(
        tbl_grid, tsol_vals, np.array(
            (rh, fmf_ret, τ_ret))).reshape(
        (16, 33))
    tsol_ret = np.empty(16)
    for i in range(16):
        tsol_ret[i] = np.interp(solz,dtheta,tsol_interp[i,:])

    tsen_interp = fast_interp(
    tbl_grid, tsen_vals, np.array(
        (rh, fmf_ret, τ_ret))).reshape(
    (16, 33))
    tsen_ret = np.empty(16)
    for i in range(16):
        tsen_ret[i] = np.interp(senz,dtheta,tsol_interp[i,:])

    return tsol_ret, tsen_ret

@njit
def get_glint(τ_ret, glint_coef, taur, solz, senz):
    airmass = 1 / np.cos(solz * np.pi / 180) + 1 / np.cos(senz * np.pi / 180)
    return glint_coef * np.exp(-(taur + τ_ret) * airmass)

# @jit

def get_angstrom(rh, fmf):
    """
    Input: rh, fmf
    output: angstrom
    """
    ξ = interp.interpn((reflut.humidity.values, reflut.fmf.values), 
        reflut.extc.values, (rh,fmf))[0,:]
    angstrom = -np.log10(ξ[1]/ξ[12]) / np.log10(443/869)
    return angstrom

#@jit
def get_toa(theta, solz, relaz, senz):
    """
    pr = theta[0]
    ws = theta[1]
    RH = theta[2]
    O3 = theta[3]
    fmf = theta[4]
    τa   = theta[5]
    """
    pr = theta[0]
    ws = theta[1]
    RH = theta[2]
    O3 = theta[3]
    fmf = theta[4]
    τa = theta[5]
    cwv = theta[6]
    chlor_a = theta[7]
    #     theta[0] = theta[0]*1000
    #     theta[1] = theta[1]*10
    #     theta[2] = theta[2]*100
    #     theta[4] = theta[4]*100
    # print(theta)
    # fmf = theta[0]
    # τa = theta[1]
    # RH = 88
    # O3 = .33
    # ws = 5
    # pr   = 1032
    # ρw = np.array([ 3.57071199e-02,  2.56224963e-02,  1.97976561e-02,  1.36691022e-02,
    #     3.18279780e-03,  1.91725941e-03,  1.52046417e-03,  4.02669929e-04,
    #     1.97751943e-05,  1.51977081e-05,  0.0,             0.0,
    #     0.0,             0.0,             0.0,             0.0])
    esd = 1.0
    rad = np.pi / F0/ np.cos(np.deg2rad(solz))
    # l, rrs = convolve_rsr('modisa', morel_rrs(chlor_a))
    rrs = morel_rrs(chlor_a, solz=0)
    brdf = morel_brdf(chlor_a, solz,senz,relaz, corr=True)
    Rrs_modis = interp.interp1d(np.arange(
        300, 701, 1), rrs, kind='nearest', bounds_error=False, fill_value="extrapolate")(wl)
    Rrs_modis[10:] = 0.0
    brdf_modis = interp.interp1d(np.arange(300,701,1), brdf, kind='nearest', bounds_error=False, fill_value="extrapolate")(wl)
    brdf_modis[10:] = 1
    taur = pr / p0 * taur_p0
    points = (solz, relaz, senz)
    sigma = 0.0731 * np.sqrt(ws)
    ray_points = (sigma, solz, senz)
    reflut_int = fast_interp(grid, vals, points).reshape((8, 10, 5, 16))
    ρr, ρrq, ρru = get_rayleigh(ray_points, relaz, pr)
    glint_coef, glitter_Q, glitter_U = getglint_iqu(
        senz, solz, relaz, ws, wind_dir=0)
    Tρg = get_glint(τa, glint_coef, taur, solz, senz)
    Tgsol = gastrans.ozone(O3, senz, solz)[0]*gastrans.h2o(cwv, senz, solz)[0]
    Tgsen = gastrans.ozone(O3, senz, solz)[1]*gastrans.h2o(cwv, senz, solz)[1]
    ρa = aer_ref(reflut_int, RH, fmf, τa)
    tsol, tsen = aer_trans(RH, fmf, τa, solz, senz)
    
    Lw = (Rrs_modis*tsol*np.cos(np.deg2rad(solz))*F0)/brdf_modis
    tLw=Lw*tsen
    Tρw=tLw*rad

    ρt = (ρr + ρa + Tρg + Tρw) * Tgsol * Tgsen
    # ρt = (ρa + Tρw)
    return ρt

def get_toa_no_w(theta, solz, relaz, senz):
    """
    pr = theta[0]
    ws = theta[1]
    RH = theta[2]
    O3 = theta[3]
    fmf = theta[4]
    τa   = theta[5]
    """
    pr = theta[0]
    ws = theta[1]
    RH = theta[2]
    O3 = theta[3]
    fmf = theta[4]
    τa = theta[5]
    cwv = theta[6]
    #     theta[0] = theta[0]*1000
    #     theta[1] = theta[1]*10
    #     theta[2] = theta[2]*100
    #     theta[4] = theta[4]*100
    # print(theta)
    # fmf = theta[0]
    # τa = theta[1]
    # RH = 88
    # O3 = .33
    # ws = 5
    # pr   = 1032
    taur = pr / p0 * taur_p0
    points = (solz, relaz, senz)
    sigma = 0.0731 * np.sqrt(ws)
    ray_points = (sigma, solz, senz)
    reflut_int = fast_interp(grid, vals, points).reshape((8, 10, 5, 16))
    ρr, ρrq, ρru = get_rayleigh(ray_points, relaz, pr)
    glint_coef, glitter_Q, glitter_U = getglint_iqu(
        senz, solz, relaz, ws, wind_dir=0)
    Tρg = get_glint(τa, glint_coef, taur, solz, senz)
    Tgsol = gastrans.ozone(O3, senz, solz)[0]*gastrans.h2o(cwv, senz, solz)[0]
    Tgsen = gastrans.ozone(O3, senz, solz)[1]*gastrans.h2o(cwv, senz, solz)[1]
    ρa = aer_ref(reflut_int, RH, fmf, τa)
    tsol, tsen = aer_trans(RH, fmf, τa, solz, senz)
    ρt = (ρr + ρa + Tρg) * Tgsol * Tgsen

    return ρt
