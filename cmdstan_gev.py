# Fitting GEV using stan

"""
Functions to use cmdstan via cmdstanpy to fit
gneralized extreme value distribution to time series data.
Calculates return levels to given return periods and
produces a plot with uncertainties.
"""

import os
import platform
import tempfile
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
from scipy.stats import genextreme

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

import cmdstanpy
from cmdstanpy import CmdStanModel

import arviz as az  # not yet used

if os.getenv('CMDSTAN') is None:
    cmdstanpy.set_cmdstan_path('/usr/local/share/cmdstan/cmdstan-2.33.1')


def gev_fit(y, hyper={}, chains=4, iter_sampling=1000):
    """MCMC fit for GEV parameters,"""
    hyper0 = {
        'mu0': np.mean(y),
        'tau0': np.std(y),
        'sig0': np.std(y),
        'sigsig0': np.std(y)*2,
        'xi0': 0.0,
        'xisig0': 1.0,
    }

    hyper0.update(hyper)

    data = {
        'y': y,
        'N': y.size,
    }
    data.update(hyper0)

    inits = {'mu': data['mu0'],
            'sigma': data['sig0'],
            'xi': data['xi0']}

    stan_file = os.path.join('stan', 'ggev.stan')
    # ei onnistu
    exe_file = os.path.join('stan', 
                            'ggev_' + platform.system() +
                            '_' + platform.machine())
    model = CmdStanModel(stan_file=stan_file,
                         # exe_file=exe_file,
                         cpp_options={'STAN_THREADS': 'true'},
                         compile=True)

    with tempfile.TemporaryDirectory() as dir:
        data_file = os.path.join(dir,'ggev_data.json')
        cmdstanpy.write_stan_json(data_file, data)

        fit = model.sample(data=data_file,
                           chains=chains,
                           iter_sampling=iter_sampling,
                           inits=inits)

    return fit


def qgev(q, loc=0, scale=1, shape=0, lowertail=True):
    """GEV quantiles as in R evd::qgev()"""
    qq = q if lowertail else 1.0 - q
    return genextreme.ppf(qq, c=-shape, loc=loc, scale=scale)


def ppoints(m, a=3/8):
    """Ordinates for probability plotting."""
    return (np.arange(1, m+1) - a) / (m + (1 - a) - a)


def gev_qpred(draws, minp=1.01, maxp=1000, n=40):
    """GEV quantiles from MCMC chain"""
    if isinstance(draws, cmdstanpy.stanfit.mcmc.CmdStanMCMC):
        draws = draws.draws_pd()
    yt = np.exp(np.log(10) *
                np.linspace(np.log10(minp),
                            np.log10(maxp), n))

    nt = yt.size
    nsimu = draws.shape[0]
    out = np.empty((nsimu, nt))

    for i in range(nsimu):
        out[i,:] = qgev(1/yt,
                        loc=draws['mu'][i],
                        scale=draws['sigma'][i],
                        shape=draws['xi'][i],
                        lowertail = False)
    mu = mquantiles(out, prob=[0.05, 0.5, 0.95], axis=0).T
    mu = pd.DataFrame(np.c_[yt, mu],
                      columns=['time', 'X0.05', 'X0.5', 'X0.95'])
    return mu


def gev_qplot(y, draws, minp=1.01, maxp=100, n=40,
              yt=None, a=3/8,
              ylabel='Return level',
              xlabel='Return period',
              title=None,
              markersize=3,
              logscale=True):
    """GEV return level plot."""
    
    if isinstance(draws, cmdstanpy.stanfit.mcmc.CmdStanMCMC):
        draws = draws.draws_pd()
    
    mu = gev_qpred(draws, minp, maxp, n)

    plt.fill_between(mu['time'], mu['X0.05'], mu['X0.95'],
                 alpha=0.4)
    plt.plot(mu['time'], mu['X0.5'], label='GEV model')
    plt.plot(1/np.flip(ppoints(y.size)), np.sort(y), 'o',
         label='Observations',
         color='black', markersize=markersize)
    if logscale:
        plt.xscale('log')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return plt.gca()


def gev_posterior(fit):
    return az.plot_posterior(fit, kind='hist',
                             hdi_prob='hide',
                             point_estimate=None)
