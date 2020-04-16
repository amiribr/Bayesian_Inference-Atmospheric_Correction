# Bayesian_Inference-Atmospheric_Correction

This project utilize probabilistic programming tools mainly **PYMC3** and **Numpyro** to infer the full posterior distribution of atmospheric and oceanic variables needed for the atmospheric correction of satellite sensors. These models were designed for MODIS sensor on-board Aqua, however it can be generalized to any other sensor.

The forward model in this work is based on a **MLP Neural Network** with 4 hidden layers with each 300 nodes. The neural network model significantly speeds up the computational time and provides analytical gradients using Backpropagation algorithm necessary for Hamiltonian Monte Carlo type samplers such as NUTS.
There are currently 4 options to run the inference:
  1. Numpyro with NUTS sampler
  2. PYMC3 with NUTS sampler
  3. PYMC3 with ADVI
  4. PYMC# with MAP

**Example output:**
1. Numpyro NUTS inference:
```
Truth parameters
           pr        ws         rh        o3        fmf         τ        wv   chlor_a       solz      relaz   senz
0  886.501099  2.033721  71.120407  0.422718  70.490639  0.227492  2.542479  6.924256  73.470001  71.849998  13.02
Initializing mean cosine of downwelling irradiance interpolation function
Initializing f-prime interpolation function
Initializing f/Q interpolation function
sample: 100%|███████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:35<00:00, 55.89it/s, 127 steps of size 3.64e-02. acc. prob=0.94]

Estimated parameters:
                mean       std    median      5.0%     95.0%     n_eff     r_hat
       FMF     69.51      3.72     69.44     64.03     76.18    504.85      1.00
        O3      0.43      0.00      0.43      0.42      0.43    945.78      1.00
        Pr    887.01      3.97    887.27    880.78    893.58    493.06      1.01
        RH     71.36      1.00     71.48     69.80     72.88    723.92      1.00
        WS      2.03      0.03      2.03      1.99      2.08   1449.31      1.00
        WV      2.54      0.04      2.54      2.48      2.59    946.11      1.00
     chlor      6.91      1.69      7.07      4.28      9.65    577.93      1.00
     relaz     71.85      0.00     71.85     71.85     71.85      0.50      1.00
      senz     13.02      0.00     13.02     13.02     13.02      0.50      1.00
      solz     73.47      0.00     73.47     73.47     73.47      0.50      1.00
        τa      0.23      0.00      0.23      0.22      0.23    535.50      1.01
```

2. PYMC3 NUTS inference:
```
Loaded Neural Network model.
building coupled model computational graph with modeling sensor noise...
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [chlor, WV, τa, FMF, WS, Pr, O3, RH]
Sampling 2 chains, 0 divergences: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [03:05<00:00, 21.58draws/s]
Estimated parameters:
          mean     sd   hpd_3%  hpd_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
RH      71.067  0.996   69.365   72.681      0.030    0.021    1092.0  1090.0    1071.0    1030.0   1.00
O3       0.422  0.005    0.413    0.431      0.000    0.000    1311.0  1311.0    1248.0     931.0   1.00
Pr     886.019  4.282  878.019  893.974      0.155    0.110     762.0   762.0     776.0    1005.0   1.01
WS       2.034  0.029    1.988    2.083      0.001    0.000    1946.0  1944.0    1857.0     919.0   1.00
FMF     70.859  3.960   63.396   77.805      0.146    0.104     734.0   725.0     764.0     950.0   1.01
τa       0.227  0.003    0.221    0.234      0.000    0.000     809.0   809.0     823.0     981.0   1.01
WV       2.544  0.036    2.484    2.601      0.001    0.001    1741.0  1741.0    1625.0    1022.0   1.00
chlor    6.521  1.685    3.740    9.711      0.056    0.040     910.0   879.0     874.0     885.0   1.00
Truth parameters:
           pr        ws         rh        o3        fmf         τ        wv   chlor_a       solz      relaz   senz
0  886.501099  2.033721  71.120407  0.422718  70.490639  0.227492  2.542479  6.924256  73.470001  71.849998  13.02
```
3. PYMC3 ADVI inference:
```
Loaded Neural Network model.
building coupled model computational graph with modeling sensor noise...
Average Loss = -74.936: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [01:24<00:00, 1178.85it/s]
Finished [100%]: Average Loss = -74.952
Estimated parameters:
          mean     sd   hpd_3%  hpd_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
RH      70.364  0.196   70.034   70.737      0.006    0.004     996.0   996.0     985.0     868.0    NaN
O3       0.422  0.003    0.417    0.427      0.000    0.000     894.0   893.0     888.0     932.0    NaN
Pr     887.302  1.312  885.023  889.891      0.039    0.027    1145.0  1145.0    1170.0     983.0    NaN
WS       2.034  0.030    1.987    2.082      0.001    0.001     909.0   909.0     936.0     988.0    NaN
FMF     68.129  0.220   67.716   68.542      0.007    0.005     959.0   959.0     956.0    1026.0    NaN
τa       0.229  0.000    0.229    0.230      0.000    0.000     908.0   908.0     903.0     836.0    NaN
WV       2.541  0.038    2.486    2.604      0.001    0.001    1048.0  1048.0    1051.0     981.0    NaN
chlor    6.637  1.245    4.095    8.733      0.040    0.028     988.0   988.0    1007.0     821.0    NaN
Truth parameters:
           pr        ws         rh        o3        fmf         τ        wv   chlor_a       solz      relaz   senz
0  886.501099  2.033721  71.120407  0.422718  70.490639  0.227492  2.542479  6.924256  73.470001  71.849998  13.02
```
4. PYMC3 MAP
```
Loaded Neural Network model.
building coupled model computational graph with modeling sensor noise...
logp = 91.421, ||grad|| = 0.55078: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 77/77 [00:00<00:00, 715.96it/s]
Estimated parameters:
           Pr        WS         RH        O3        FMF        τa        WV     chlor
0  886.519821  2.034128  71.096377  0.422761  70.432002  0.227518  2.542739  6.924603
Truth parameters:
           pr        ws         rh        o3        fmf         τ        wv   chlor_a       solz      relaz   senz
0  886.501099  2.033721  71.120407  0.422718  70.490639  0.227492  2.542479  6.924256  73.470001  71.849998  13.02
```
