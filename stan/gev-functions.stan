//Common GEV functions for all model versions
//gev_lpdf calculates the log-likelihood given observations
//gev_vector is a vectorised version used for cross validation purposes
//gev_rng generates GEV-distributed random numbers for given parameter values 

functions {

  real gev_lpdf(vector y, real mu, real sigma, real xi) {
    vector[rows(y)] z;
    vector[rows(y)] lp;
    int N;
    N = rows(y);
    if (abs(xi) > 1e-15) {
      z =   1 + xi * ((y - mu) / sigma);
      for(n in 1:N){
	z[n] = pow(z[n],-1/xi);
      }
    } else {
      z =  exp(-(y - mu)/ sigma);
    }
    lp = log(sigma) - (1 + xi) * log(z) + z;
    return -sum(lp);
    }
  
  vector gev_vector(vector y, real mu, real sigma, real xi) {
    vector[rows(y)] z;
    vector[rows(y)] lp;
    int N;
    N = rows(y);
    // assert rows(mu) == N;
    if (abs(xi) > 1e-15) {
      z =  1 + xi * ((y - mu) / sigma);
      for(n in 1:N){
        z[n] = pow(z[n],-1/xi);
      }
    } else {
      z = exp(-(y - mu)/ sigma);
    }
    lp = log(sigma) - (1 + xi) * log(z) + z;
    return -lp;
  } 
  
  real gev_rng(real mu, real sigma, real xi) {
   real x;
   if(sigma <= 0){
     reject("Rejecting, sigma=", sigma);
   }
   if(abs(xi) > 1e-15){
     x = mu + (sigma / xi) * (pow(-log(uniform_rng(0,1)),-xi) - 1);
   } else {
     x = mu - log(-log(uniform_rng(0,1))) * sigma;
   }
   return x;
  }
}

