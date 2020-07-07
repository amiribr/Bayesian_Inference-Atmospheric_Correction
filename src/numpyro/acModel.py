import os
import pickle
import jax.numpy as np
import numpy as onp
import pandas as pd
import numpyro
import numpyro.distributions as dist
from forward_model import fwModel
prj_path = os.environ['BAYES_PRJ']

NN_path = os.path.join(prj_path, 'data/NN_model/NN_fwd_mdl.h5')
scaler_path = os.path.join(
    prj_path, 'data/NN_model/best_model_pca_tf_lognorm_input_scaler.bin')
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

    pr_min = onp.maximum(pr - pr*pr_unc, 800)
    pr_max = onp.minimum(pr + pr*pr_unc, 1080)

    ws_min = onp.maximum(ws - ws*ws_unc, 0.1)
    ws_max = onp.minimum(ws + ws*ws_unc, 15)

    RH_min = onp.maximum(rh - rh*RH_unc, 30)
    RH_max = onp.minimum(rh + rh*RH_unc, 95)

    O3_min = onp.maximum(o3 - o3*O3_unc, 0.2)
    O3_max = onp.minimum(o3 + o3*O3_unc, 0.45)

    wv_min = onp.maximum(wv - wv*wv_unc, 0)
    wv_max = onp.minimum(wv + wv*wv_unc, 10)

    # RH = numpyro.sample('RH', dist.TruncatedNormal(low=0, loc=rh, scale=1))
    # O3 = numpyro.sample('O3', dist.TruncatedNormal(low=0, loc=o3, scale=1))
    # Pr = numpyro.sample('Pr', dist.TruncatedNormal(low=0, loc=pr, scale=1))
    # WS = numpyro.sample('WS', dist.TruncatedNormal(low=0, loc=ws, scale=1))
    # FMF = numpyro.sample('FMF', dist.Uniform(0, 100))
    # τa = numpyro.sample('τa', dist.Uniform(0.01, 0.4))
    # WV = numpyro.sample('WV', dist.TruncatedNormal(low=0, loc=wv, scale=1))
    # chlor = numpyro.sample('chlor', dist.Uniform(0.001, 10))
    # solz = numpyro.sample('solz', dist.Uniform(solz, solz))
    # relaz = numpyro.sample('relaz', dist.Uniform(relaz, relaz))
    # senz = numpyro.sample('senz', dist.Uniform(senz, senz))
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
