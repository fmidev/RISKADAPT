# Fitting GEV using stan

"""
Functions to use cmdstan via cmdstanpy to fit
gneralized extreme value distribution to time series data.
Calculates return levels to given return periods and
produces a plot with uncertainties.

xgev_fit calculates return levels for each pixel in xarray DataSet
"""

import os
import platform
import tempfile
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
from scipy.stats import genextreme

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

import cmdstanpy
from cmdstanpy import CmdStanModel

import arviz as az  # for plotting

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")

if os.getenv('CMDSTAN') is None:
    cmdstanpy.set_cmdstan_path('/usr/local/share/cmdstan/cmdstan-2.33.1')


def gev_fit(y, hyper={}, chains=4, iter_sampling=1000, quiet=None,
            force_compile=False, **kwargs):
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
    # Would like to have different exe for different architectures
    exe_file = os.path.join('stan', 
                            'ggev_' + platform.system() +
                            '_' + platform.machine())

    model = CmdStanModel(stan_file=stan_file,
                         # exe_file=exe_file,
                         cpp_options={'STAN_THREADS': 'true'},
                         #cpp_options={'STAN_THREADS': 'true', 'STAN_CPP_OPTIMS': 'true'},
                         #cpp_options={'STAN_THREADS': 'true',
                         #             'STAN_MPI': 'true',
                         #             'CXX': 'mpicxx',
                         #             'TBB_CXX_TYPE': 'clang',
                         #             'STAN_CPP_OPTIMS': 'true'},
                         #force_compile=force_compile # in cmdstanpy 1.2.0
                         )

    if quiet is not None:
        cmdstanpy_logger.disabled = quiet

    with tempfile.TemporaryDirectory() as dir:
        data_file = os.path.join(dir, 'ggev_data.json')
        cmdstanpy.write_stan_json(data_file, data)
        fit = model.sample(data=data_file,
                           chains=chains,
                           iter_sampling=iter_sampling,
                           inits=inits,
                           **kwargs)

    return fit


def qgev(q, loc=0, scale=1, shape=0, lowertail=True):
    """GEV quantiles as in R evd::qgev()"""
    qq = q if lowertail else 1.0 - q
    return genextreme.ppf(qq, c=-shape, loc=loc, scale=scale)


def ppoints(m, a=3/8):
    """Ordinates for probability plotting."""
    return (np.arange(1, m + 1) - a) / (m + (1 - a) - a)


def gev_qpred(draws, minp=1.01, maxp=1000, n=40, yt=None,
              prob=[0.05, 0.5, 0.95], return_chain=False):
    """GEV quantiles from MCMC chain"""
    if isinstance(draws, cmdstanpy.stanfit.mcmc.CmdStanMCMC):
        draws = draws.draws_pd()
    if yt is None:
        yt = np.exp(np.log(10) *
                    np.linspace(np.log10(minp),
                                np.log10(maxp), n))

    yt = np.atleast_1d(yt)
    nt = yt.size
    nsimu = draws.shape[0]
    out = np.empty((nsimu, nt))

    for i in range(nsimu):
        out[i,:] = qgev(1/yt,
                        loc=draws['mu'][i],
                        scale=draws['sigma'][i],
                        shape=draws['xi'][i],
                        lowertail = False)
    if return_chain:
        return pd.DataFrame(out, columns=yt)
    mu = mquantiles(out, prob=prob, axis=0).T
    mu = pd.DataFrame(np.c_[yt, mu],
                      columns=['return_period',]+['X'+str(s) for s in prob])
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

    plt.fill_between(mu['return_period'], mu['X0.05'], mu['X0.95'],
                 alpha=0.4)
    plt.plot(mu['return_period'], mu['X0.5'], label='GEV model')
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


def _xgev0(y, yt, hyper={}, prob=[0.05, 0.5, 0.95], dtype=np.float32):
    try:
        fit = gev_fit(y, hyper=hyper, adapt_delta=0.85,
                      show_progress=False, show_console=False, quiet=True)
    except Exception as ex:
        logging.warning(ex)
        mu = xr.DataArray(np.empty((len(prob), len(yt)), dtype=dtype)*np.nan,
                          coords=[('quantile', prob), ('return_period', yt.values)])
        return mu
    mu = gev_qpred(fit, yt=yt, prob=prob)
    mu = mu.set_index('return_period')
    mu = mu.to_xarray().to_array().rename({'variable': 'quantile'}).astype(dtype)
    return mu


def xgev_fit(da, yt, prob=[0.05, 0.5, 0.95], timevar='time', keep_attrs=True,
             dtype=np.float32):
    """
    GEV fit over 'time' for each pixel in an xarray DataSet.
    """
    xyt = xr.DataArray(yt, coords=[('return_period', yt)])
    res = xr.apply_ufunc(
        _xgev0, da, xyt, kwargs={'prob': prob, 'dtype': dtype},
        input_core_dims=[[timevar], ['return_period']],
        output_core_dims=[['quantile', 'return_period']],
        keep_attrs=keep_attrs,
        dask='parallelized',
        output_dtypes=[dtype],
        vectorize=True,  # _xgev0 is not vectorized
    )
    res = res.assign_coords({'quantile': prob})
    return res
