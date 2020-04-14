import os
import pickle
import jax.numpy as np
import numpy as onp
import pandas as pd
import numpyro
import numpyro.distributions as dist
from forward_model import fwModel

NN_path = '../../data/NN_model/best_model_pca_tf_lognorm/'
scaler_path = '../../data/NN_model/best_model_pca_tf_lognorm_input_scaler.bin'
fwd = fwModel(NN_path, scaler_path)
fwd.load_NN(scaler_path)
forward_model = fwd.forward_model

def acModel(ρ_obs, priors, σ_mdl, nir):

    pr = priors[0]
    ws = priors[1]
    rh = priors[2]
    o3 = priors[3]
    wv = priors[4]
    solz = priors[5]
    relaz = priors[6]
    senz = priors[7]
    pr_unc = 0.025
    ws_unc = 0.025
    RH_unc = 0.025
    O3_unc = 0.025
    wv_unc = 0.025

    pr_min = pr - pr*pr_unc
    pr_max = pr + pr*pr_unc

    ws_min = onp.maximum(ws - ws*ws_unc, 0)
    ws_max = ws + ws*ws_unc

    RH_min = rh - rh*RH_unc
    RH_max = rh + rh*RH_unc

    O3_min = o3 - o3*O3_unc
    O3_max = o3 + o3*O3_unc

    wv_min = wv - wv*wv_unc
    wv_max = wv + wv*wv_unc
    RH = numpyro.sample('RH', dist.Uniform(RH_min, RH_max))
    O3 = numpyro.sample('O3', dist.Uniform(O3_min, O3_max))
    Pr = numpyro.sample('Pr', dist.Uniform(pr_min, pr_max))
    WS = numpyro.sample('WS', dist.Uniform(ws_min, ws_max))
    FMF = numpyro.sample('FMF', dist.Uniform(0, 100))
    τa = numpyro.sample('τa', dist.Uniform(0.01, 0.4))
    WV = numpyro.sample('WV', dist.Uniform(wv_min, wv_max))
    chlor = numpyro.sample('chlor', dist.Uniform(0.001, 10))
    solz = numpyro.sample('solz', dist.Uniform(solz, solz))
    relaz = numpyro.sample('relaz', dist.Uniform(relaz, relaz))
    senz = numpyro.sample('senz', dist.Uniform(senz, senz))

    inputs = (((Pr, WS, RH, O3, FMF, τa, WV, chlor, solz, relaz, senz)))
    idx = 0
    if nir:
        idx = -3
    μ = (forward_model(np.array(inputs)))[idx:]
    numpyro.sample('ρ_pred', dist.Normal(μ, σ_mdl[idx:]), obs=ρ_obs[idx:])
