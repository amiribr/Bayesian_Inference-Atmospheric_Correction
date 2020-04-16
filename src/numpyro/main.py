import pickle
from forward_model import fwModel 
import sys
sys.path.insert(
    0, '/Users/aibrahi2/Research/Bayesian_Inference-ML-Atmospheric_Correction/src')
import LUT_generator.ac_likelihood_w_chlor as ac_likelihood_w_chlor
from acModel import acModel
import numpyro
from numpyro.infer import MCMC, NUTS
import numpy as onp
from scipy.interpolate import RegularGridInterpolator as RGI
import argparse
import jax.random as random
import pandas as pd

NN_path = '../../data/NN_model/NN_fwd_mdl.h5'
scaler_path = '../../data/NN_model/best_model_pca_tf_lognorm_input_scaler.bin'
fwd = fwModel(NN_path, scaler_path)
fwd.load_NN(scaler_path)
forward_model = fwd.forward_model

def σ_per_geom(solz, relaz, senz):
    diff_per_geom = onp.load('../../data/NN_model/diff_per_geom.npy')
    θ0 = onp.linspace(0, 80, 9)
    ϕ = onp.linspace(0, 180, 19)
    θ = onp.linspace(0, 70, 8)
    return RGI((θ0, ϕ, θ), diff_per_geom)((solz, relaz, senz))

def run_inference(model, args, rng_key, priors, geom, observations, nir):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains,
                progress_bar=True, chain_method='parallel')
    solz = geom[0]; relaz = geom[1]; senz = geom[2]
    σ_mdl = σ_per_geom(solz, relaz, senz)
    mcmc.run(rng_key, observations, priors, σ_mdl, nir)
    mcmc.print_summary()
    trace = mcmc.get_samples(group_by_chain=True)
    return trace


def main(args, priors, geom, observation, nir):
    # do inference
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    samples = run_inference(acModel, args, rng_key_,
                                priors, geom, observation, nir)
    return samples


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description="Atmospheric Correction Inference")
    parser.add_argument("-n", "--num_samples", nargs="?",
                        default=1000, type=int)
    parser.add_argument("--num_warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num_chains", nargs='?', default=1, type=int)
    parser.add_argument("--device", default='cpu',
                        type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


    with open('/Users/aibrahi2/Research/atmocor/chlor_based_ret/in_out_na_df.pkl', 'rb') as pickle_file:
        in_out_na_df = pickle.load(pickle_file)

    NN = 0
    LUT = 1
    nir = 0

    for i in range(500,501):
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
        labels = ['pr', 'ws', 'rh', 'o3', 'fmf', 'τ', 'wv', 'chlor_a', 'solz', 'relaz', 'senz']
        print('Truth parameters')
        print(pd.DataFrame(data=onp.atleast_2d(onp.array(inputs_)), columns=labels))

        if NN:
            ρ_obs = (forward_model(inputs_))[0:13]
        elif LUT:
            ρ_obs = ac_likelihood_w_chlor.get_toa(
                onp.array([pr, ws, rh, o3, fmf, τ, wv, chlor_a]), solz, relaz, senz)[0, 0:13]
    geom = (solz, relaz, senz)
    main(args, priors, geom, ρ_obs, nir)
