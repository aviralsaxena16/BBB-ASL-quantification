functions {
  vector deltaM_model(vector t, real att, real f, real M0a, real tau, 
                     real T1a, real T1, real lambd, real a) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    real T1app = 1.0 / (1.0 / T1 + f / lambd);
    real R = 1.0 / T1app - 1.0 / T1a;

    for (i in 1:n) {
      if (t[i] >= att && t[i] <= att + tau) {
        // Case 2: att <= t <= att + tau
        real time = t[i];
        real term = exp(R * time) - exp(R * att);
        deltaM[i] = (2 * a * M0a * f * exp(-time / T1app) / R) * term;
      } else if (t[i] > att + tau) {
        // Case 3: t > att + tau
        real time = t[i];
        real term = exp(R * (att + tau)) - exp(R * att);
        deltaM[i] = (2 * a * M0a * f * exp(-time / T1app) / R) * term;
      }
      // else: t < att, deltaM[i] remains 0
    }
    return deltaM;
  }
}

data {
  int<lower=0> n;
  vector[n] t;
  vector[n] signal;
  real<lower=0> M0a;
  real<lower=0> tau;

  // Fixed parameter values
  real<lower=0> T1a_fixed;
  real<lower=0> T1_fixed;
  real<lower=0> lambd_fixed;
  real<lower=0> a_fixed;

  // Fitting flags
  int<lower=0,upper=1> fit_T1a;
  int<lower=0,upper=1> fit_T1;
  int<lower=0,upper=1> fit_lambd;

  // Priors
  real T1a_prior_mean;
  real T1a_prior_std;
  real T1_prior_mean;
  real T1_prior_std;
  real lambd_prior_mean;
  real lambd_prior_std;

  // ATT and CBF priors from LS fitting
  int<lower=0,upper=1> use_att_prior_from_ls;
  int<lower=0,upper=1> use_cbf_prior_from_ls;
  real att_prior_from_ls;
  real att_prior_std;
  real cbf_prior_from_ls;
  real cbf_prior_std;

  // Bounds
  real T1a_lower;
  real T1a_upper;
  real T1_lower;
  real T1_upper;
  real lambd_lower;
  real lambd_upper;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=0.001, upper=0.2> f;  // CBF as f parameter
  real<lower=0.001> sigma;

  // Conditional parameters
  real<lower=T1a_lower, upper=T1a_upper> T1a_param;
  real<lower=T1_lower, upper=T1_upper> T1_param;
  real<lower=lambd_lower, upper=lambd_upper> lambd_param;
}

transformed parameters {
  real T1a_use;
  real T1_use;
  real lambd_use;
  real cbf;  // CBF in ml/min/100g

  // Use fitted or fixed values
  T1a_use = fit_T1a ? T1a_param : T1a_fixed;
  T1_use = fit_T1 ? T1_param : T1_fixed;
  lambd_use = fit_lambd ? lambd_param : lambd_fixed;

  // Convert f to CBF
  cbf = f * 6000.0;
}

model {
  vector[n] mu;

  if (use_att_prior_from_ls == 1) {
    att ~ normal(att_prior_from_ls, att_prior_std);
  } else {
    att ~ normal(1.2, 0.5);
  }

  if (use_cbf_prior_from_ls == 1) {
    f ~ normal(cbf_prior_from_ls / 6000.0, cbf_prior_std / 6000.0);
  } else {
    f ~ normal(0.01, 0.0025);  // Default: 60 ml/min/100g ± 15
  }

  sigma ~ exponential(1);

  // Conditional priors for flexible parameters
  if (fit_T1a) {
    T1a_param ~ normal(T1a_prior_mean, T1a_prior_std);
  }
  if (fit_T1) {
    T1_param ~ normal(T1_prior_mean, T1_prior_std);
  }
  if (fit_lambd) {
    lambd_param ~ normal(lambd_prior_mean, lambd_prior_std);
  }

  // Likelihood
  mu = deltaM_model(t, att, f, M0a, tau, T1a_use, T1_use, lambd_use, a_fixed);
  signal ~ normal(mu, sigma);
}