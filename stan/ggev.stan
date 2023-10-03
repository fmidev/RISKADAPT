/*

Simple GEV fit in stan

*/
 
#include gev-functions.stan

data {
  int<lower=0> N;
  vector[N] y;
  real mu0;
  real<lower=0> tau0;
  real<lower=0> sig0;
  real<lower=0> sigsig0;
  real xi0;
  real<lower=0> xisig0;
}

parameters {
  real mu;
  real<lower=0> sigma;  
  real<lower=-2.0, upper=2.0> xi;
}

model {
  mu ~ normal(mu0, tau0);
  sigma ~ normal(sig0, sigsig0);
  xi ~ normal(xi0, xisig0);
  y ~ gev(mu, sigma, xi);
}

