#!/usr/bin/env python

from joblib import dump, load
import theano.tensor as tt
from theano import shared as shared
from theano import dot as dot
import pymc3 as pm
import numpy as np
import warnings
import pickle

warnings.filterwarnings("ignore", category=UserWarning)

wl = np.array([412.,  443.,  469.,  488.,  531.,  547.,  555.,  645.,  667.,
               678.,  748.,  859.,  869., 1240., 1640., 2130.])


def get_sigma(ρt, solz):

    F0 = 10*np.array([172.912, 187.622, 205.878, 194.933, 185.747,
                      186.539, 183.869, 157.811, 152.255, 148.052,
                      128.065, 97.174, 95.824, 45.467, 23.977, 9.885])
    rad = np.pi/F0[0:13]/np.cos(np.deg2rad(solz))
    C0 = np.array([0.05499859, 0.0293947, 0.11931482, 0.01927545, 0.01397522,
                   0.01139088, 0.08769538, 0.10406925, 0.00496291, 0.00427147,
                   0.00416994, 0.004055895, 0.00312263, 0.07877732, 0.01743281,
                   0.00628912])
    C1 = np.array([0.0000834, 0.0000938, 0.00008195, 0.0000945, 0.0001004,
                   0.0001648, 0.00007, 0.00008533, 0.0001405, 0.0001316,
                   0.0002125, 0.00019755, 0.000186, 0.0004994, 0.001044864,
                   0.0002116])
    Lt = ρt/rad
    noise = C0[0:13] + C1[0:13] * Lt
    snr = Lt / noise
    sigma = 1 / snr
    return sigma/10


def acInferModel(observations, forward_model, fwd_mdl_unc, anc, anc_unc,
                 geom, prior_dist, model_noise=True, coupled=True):
    with open('/Users/aibrahi2/Research/atmocor/chlor_based_ret/in_out_na_df.pkl', 'rb') as pickle_file:
        inp = pickle.load(pickle_file)

    dtype = "float64"
    # Initialize vars as theano tensors for prior dist
    pr_min = tt.dscalar(dtype)
    pr_max = tt.dscalar(dtype)

    ws_min = tt.dscalar(dtype)
    ws_max = tt.dscalar(dtype)

    RH_min = tt.dscalar(dtype)
    RH_max = tt.dscalar(dtype)

    O3_min = tt.dscalar(dtype)
    O3_max = tt.dscalar(dtype)

    wv_min = tt.dscalar(dtype)
    wv_max = tt.dscalar(dtype)

    pr = anc[0]
    ws = anc[1]
    rh = anc[2]
    o3 = anc[3]
    wv = anc[4]
    pr_unc = anc_unc[0]
    ws_unc = anc_unc[1]
    RH_unc = anc_unc[2]
    O3_unc = anc_unc[3]
    wv_unc = anc_unc[4]

    solz = tt.dscalar(dtype)
    relaz = tt.dscalar(dtype)
    senz = tt.dscalar(dtype)
    solz = (geom[0])
    relaz = (geom[1])
    senz = (geom[2])
    
    const = 3.4
    # const = 1e-5

    pr_min = tt.maximum(pr - pr*pr_unc, inp['pr'].min()).eval()
    pr_max = tt.minimum(pr + pr*pr_unc, inp['pr'].max()).eval()
    pr_std = (pr_max - pr_min)/const

    ws_min = tt.maximum(ws - ws*ws_unc, inp['ws'].min()).eval()
    ws_max = tt.minimum(ws + ws*ws_unc, inp['ws'].max()).eval()
    ws_std = (ws_max - ws_min)/const

    RH_min = tt.maximum(rh - rh*RH_unc, inp['rh'].min()).eval()
    RH_max = tt.minimum(rh + rh*RH_unc, inp['rh'].max()).eval()
    rh_std = (RH_max - RH_min)/const

    O3_min = tt.maximum(o3 - o3*O3_unc, inp['o3'].min()).eval()
    O3_max = tt.minimum(o3 + o3*O3_unc, inp['o3'].max()).eval()
    O3_std = (O3_max - O3_min)/const

    wv_min = tt.maximum(wv - wv*wv_unc, inp['wv'].min()).eval()
    wv_max = tt.minimum(wv + wv*wv_unc, inp['wv'].max()).eval()
    wv_std = (wv_max - wv_min)/const

    print('pr_std: ', pr_std, ' ws_std: ', ws_std, ' rh_std: ', rh_std,' o3_std: ', O3_std, ' wv_std: ', wv_std)
     
    if fwd_mdl_unc.any() == None:
        fwd_mdl_unc = 0.00016
    with pm.Model() as model:
        if prior_dist == 'normal':
            RH = pm.Normal('RH', mu=rh, sigma=rh_std, dtype = dtype)
            O3 = pm.Normal('O3', mu=o3, sigma=O3_std, dtype = dtype)
            Pr = pm.Normal('Pr', mu=pr, sigma=pr_std, dtype = dtype)
            WS = pm.Normal('WS', mu=ws, sigma=ws_std, dtype = dtype)
            WV = pm.Normal('WV', mu=wv, sigma=wv_std, dtype = dtype)
            FMF = pm.Normal('FMF', mu=50, sigma=50, dtype=dtype)
            τa = pm.Normal('τa', mu=0.2, sigma=1, dtype=dtype)
            chlor = pm.Normal('chlor', mu=5, sigma=10, dtype=dtype)

        if prior_dist == 'bnormal':
            RH_bound = pm.Bound(pm.Normal, lower=inp['rh'].min(), upper=inp['rh'].max())
            RH = RH_bound('RH', mu=rh, sigma=rh_std, dtype = dtype)
            O3_bound = pm.Bound(pm.Normal, lower=inp['o3'].min(), upper=inp['o3'].max())
            O3 = O3_bound('O3', mu=o3, sigma=O3_std, dtype = dtype)
            pr_bound = pm.Bound(pm.Normal, lower=inp['pr'].min(), upper=inp['pr'].max())
            Pr = pr_bound('Pr', mu=pr, sigma=pr_std, dtype = dtype)
            WS_bound = pm.Bound(pm.Normal, lower=inp['ws'].min(), upper=inp['ws'].max())
            WS = WS_bound('WS', mu=ws, sigma=ws_std, dtype = dtype)
            WV_bound = pm.Bound(pm.Normal, lower=inp['wv'].min(), upper=inp['wv'].max())
            WV = WV_bound('WV', mu=wv, sigma=wv_std, dtype = dtype)
            # O3 = pm.Normal('O3', mu=o3, sigma=O3_std, dtype=dtype)
            # Pr = pm.Normal('Pr', mu=pr, sigma=pr_std, dtype=dtype)
            # WS = pm.Normal('WS', mu=ws, sigma=ws_std, dtype=dtype)
            FMF = pm.Uniform('FMF', lower=0, upper=100, dtype=dtype)
            τa = pm.Uniform('τa', lower=0.01, upper=0.4, dtype=dtype)
            # WV = pm.Normal('WV', mu=wv, sigma=wv_std, dtype=dtype)
            chlor = pm.Uniform('chlor', lower=0.001, upper=10, dtype=dtype)

        elif prior_dist == 'uniform':
            RH = pm.Uniform('RH', lower=RH_min, upper=RH_max, dtype=dtype)
            O3 = pm.Uniform('O3', lower=O3_min, upper=O3_max, dtype=dtype)
            Pr = pm.Uniform('Pr', lower=pr_min, upper=pr_max, dtype=dtype)
            WS = pm.Uniform('WS', lower=ws_min, upper=ws_max, dtype=dtype)
            FMF = pm.Uniform('FMF', lower=0, upper=100, dtype=dtype)
            τa = pm.Uniform('τa', lower=0.01, upper=0.4, dtype=dtype)
            WV = pm.Uniform('WV', lower=wv_min, upper=wv_max, dtype=dtype)
            chlor = pm.Uniform('chlor', lower=0.001, upper=10, dtype=dtype)
        
        inputs = [Pr, WS, RH, O3, FMF, τa, WV, chlor,
                    solz, relaz, senz]
        if coupled:
            if model_noise:
                print(
                    'building coupled model computational graph with modeling sensor noise...')
                # print(get_sigma(observations, solz))
                sensor_noise = get_sigma(observations, solz)
                total_unc = np.sqrt(sensor_noise**2 + fwd_mdl_unc**2)
                # model_obs = pm.Normal(
                #     'mobs', observations, get_sigma(observations, solz), shape=len(observations))
                # obs = pm.Deterministic(
                #     "obs", observations + model_obs * get_sigma(observations, solz))
                ρ_pred = pm.Normal('ρ_pred',
                                   mu=(forward_model(inputs))[
                                       0:len(observations)],
                                   sd=total_unc,
                                   observed=observations)
            else:
                print(
                    'building coupled model computational graph without modeling sensor noise...')
                ρ_pred = pm.Normal('ρ_pred',
                                   mu=(forward_model(inputs))[
                                       0:13],
                                   sd=fwd_mdl_unc,
                                   observed=observations)
        else:
            nir_indx = wl[0:13] > 700
            observations_ = observations[nir_indx]
            if model_noise:
                print('building model computational graph with modeling sensor noise...')
                model_obs = pm.Normal('mobs', 0, 1, shape=len(observations_))
                obs = pm.Deterministic(
                    "obs", observations_ + model_obs * get_sigma(observations, solz)[nir_indx])
                ρ_pred = pm.Normal('ρ_pred',
                                   mu=(forward_model(inputs))[
                                       0:len(observations_)],
                                   sd=fwd_mdl_unc,
                                   observed=obs)
            else:
                print(
                    'building model computational graph without modeling sensor noise...')
                ρ_pred = pm.Normal('ρ_pred',
                                   mu=(forward_model(inputs))[
                                       0:len(observations)],
                                   sd=fwd_mdl_unc,
                                   observed=observations)
    return model



class PyMCModel:
    def __init__(self, model, **model_kws):
        self.model = model

    def fit(self, n_samples=2000, **sample_kws):
        with self.model:
            self.trace_ = pm.sample(n_samples, **sample_kws)

    def fit_ADVI(self, n_samples=2000, n_iter=100000, inference='advi', **fit_kws):
        with self.model:
            self.approx_fit = pm.fit(n=n_iter, method=inference, **fit_kws)
            self.trace_ = self.approx_fit.sample(draws=n_samples)

    def fit_MAP(self):
        with self.model:
            map_estimate = pm.find_MAP(method = 'TNC')
            # Hess = pm.find_hessian(map_estimate, vars=[self.model.RH, 
            #     self.model.O3, self.model.Pr, self.model.WS, self.model.FMF, 
            #     self.model.τa, self.model.WV, self.model.chlor], model=self.model)
            Hess = pm.find_hessian(map_estimate, model=self.model)
            cov = np.linalg.inv(Hess)
        return map_estimate, cov
    
    def summary(self, show_feats):
        return pm.summary(self.trace_, varnames=show_feats)

    def show_model(self, save=False, view=True, cleanup=True):
        model_graph = pm.model_to_graphviz(self.model)
        if save:
            model_graph.render(save, view=view, cleanup=cleanup)
        if view:
            return model_graph

    def predict(self, **ppc_kws):
        ppc_ = pm.sample_ppc(self.trace_, model=self.model, **ppc_kws)
        return ppc_

    def get_waic(self):
        return pm.waic(trace=self.trace_, model=self.model)

    def get_loo(self):
        return pm.loo(trace=self.trace_, model=self.model)

    def evaluate_fit(self, show_feats):
        return pm.traceplot(self.trace_, varnames=show_feats)

    def show_forest(self, show_feats, feat_labels=None):
        g = pm.forestplot(self.trace_, varnames=show_feats,
                          ylabels=feat_labels)
        f = pl.gcf()
        try:
            ax = f.get_axes()[1]
        except IndexError:
            ax = f.get_axes()[0]
        ax.grid(axis='y')
        return g
