#!/usr/bin/env python
import pandas as pd
import argparse
import sys
import os
prj_path = os.environ['BAYES_PRJ']
sys.path.insert(
    0, os.path.join(prj_path, 'src'))
from forward_model import fwModel
from scipy.interpolate import RegularGridInterpolator as RGI
from acModel import PyMCModel
from acModel import acInferModel
import pickle
import numpy as np
import pymc3 as pm

NN_path = os.path.join(prj_path, 'data/NN_model/NN_fwd_mdl.h5')
scaler_path = os.path.join(
    prj_path, 'data/NN_model/best_model_pca_tf_lognorm_input_scaler.bin')
fwd = fwModel(NN_path, scaler_path)
fwd.load_NN(scaler_path)
forward_model = fwd.forward_model


def σ_per_geom(solz, relaz, senz):
    file_path = os.path.join(prj_path, 'data/NN_model/diff_per_geom.npy')
    diff_per_geom = np.load(file_path)
    θ0 = np.linspace(0, 80, 9)
    ϕ = np.linspace(0, 180, 19)
    θ = np.linspace(0, 70, 8)
    return RGI((θ0, ϕ, θ), diff_per_geom)((solz, relaz, senz))


def run_inference(observations, args, priors, priors_unc, geom):
    solz = geom[0]
    relaz = geom[1]
    senz = geom[2]
    σ_mdl = σ_per_geom(solz, relaz, senz)
    pm_mdl = acInferModel(observations, forward_model, σ_mdl, priors, np.array(priors_unc),
                        geom, args.prior_dist, args.model_noise, args.coupled)
    
    if args.inference_method == 'NUTS':
        PyMCModel.fit(pm_mdl, tune=args.num_warmup,
                    n_samples=args.num_samples, chains=args.num_chains)
        print('Estimated parameters:')
        print(PyMCModel.summary(pm_mdl, show_feats=None))

    elif args.inference_method == 'ADVI':
        PyMCModel.fit_ADVI(pm_mdl, n_samples=args.num_samples)
        print('Estimated parameters:')
        print(PyMCModel.summary(pm_mdl, show_feats=None))
    
    elif args.inference_method == 'MAP':
        map_approx = PyMCModel.fit_MAP(pm_mdl)
        map_df = pd.DataFrame(map_approx, index=[0])
        labels = ['Pr', 'WS', 'RH', 'O3', 'FMF', 'τa',
                  'WV', 'chlor']
        print('Estimated parameters:')
        print(pd.DataFrame(map_df.loc[0, labels]).T)

    else:
        print('Choose inference method between NUTS, ADVI or MAP')
        exit()


def main(observations, args, priors, priors_unc, geom,):

    run_inference(observations, args, priors,
                  priors_unc, geom)

if __name__ == "__main__":

    assert pm.__version__.startswith('3.9.2')

    parser = argparse.ArgumentParser(
        description="Atmospheric Correction Inference")
    parser.add_argument("-n", "--num_samples", nargs="?",
                        default=1000, type=int)
    parser.add_argument("--num_warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num_chains", nargs='?', default=1, type=int)
    parser.add_argument("--model_noise", default='1',
                        type=int, help='use "1" for True or "0" for false.')
    parser.add_argument("--coupled", default='1',
                        type=int, help='use "1" for use all wavelength or "0" for nir only.')
    parser.add_argument("--prior_dist", default='uniform',
                        type=str, help='either uniform or normal')
    parser.add_argument("--inference_method", default='NUTS',
                        type=str, help='choose between NUTS, ADVI, or MAP')

    args = parser.parse_args()

    pkl_path = os.path.join(prj_path, 'data/in_out_na_df.pkl')
    with open(pkl_path, 'rb') as pickle_file:
        in_out_na_df = pickle.load(pickle_file)


    NN = 0
    LUT = 1
    nir = 0

    for i in range(502,503):
        pr = in_out_na_df['pr'].values[i]
        ws = in_out_na_df['ws'].values[i]
        rh = in_out_na_df['rh'].values[i]
        o3 = in_out_na_df['o3'].values[i]
        fmf = in_out_na_df['fmf'].values[i]
        τ = in_out_na_df['τ'].values[i]
        wv = in_out_na_df['wv'].values[i]
        chlor_a = in_out_na_df['chlor_a'].values[i]
        solz = in_out_na_df['solz'].values[i]
        relaz = in_out_na_df['relaz'].values[i]
        senz = in_out_na_df['senz'].values[i]

        priors = [pr, ws, rh, o3, wv, solz, relaz, senz]
        inputs_ = [pr, ws, rh, o3, fmf, τ, wv, chlor_a, solz, relaz, senz]


        if NN:
            ρ_obs = (forward_model(inputs_))[0:13].eval()
        elif LUT:
            from LUT_generator import ac_likelihood_w_chlor
            ρ_obs = ac_likelihood_w_chlor.get_toa(
                np.array([pr, ws, rh, o3, fmf, τ, wv, chlor_a]), solz, relaz, senz)[0, 0:13]
    geom = (solz, relaz, senz)
    priors_unc = [0.025, 0.025, 0.025, 0.025, 0.025]
    main(ρ_obs, args, priors, priors_unc, geom)
    labels = ['pr', 'ws', 'rh', 'o3', 'fmf', 'τ',
              'wv', 'chlor_a', 'solz', 'relaz', 'senz']
    print('Truth parameters:')
    print(pd.DataFrame(data=np.atleast_2d(
        np.array(inputs_)), columns=labels))
