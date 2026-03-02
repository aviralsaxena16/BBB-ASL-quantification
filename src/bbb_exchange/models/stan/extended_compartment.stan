functions {
  vector deltaM_model_ext(vector t, real att, real f, real M0a, real tau, 
                          real abv, real att_a, real T1a, real T1, real lambd, real a) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    // Tissue compartment (Eq. 1)
    real T1app = 1.0 / (1.0 / T1 + f / lambd);
    real R = 1.0 / T1app - 1.0 / T1a;

    for (i in 1:n) {
      // Tissue
      if (t[i] >= att && t[i] <= att + tau) {
        real term = exp(R * t[i]) - exp(R * att);
        deltaM[i] += (2 * a * M0a * f * exp(-t[i] / T1app) / R) * term;
      } else if (t[i] > att + tau) {
        real term = exp(R * (att + tau)) - exp(R * att);
        deltaM[i] += (2 * a * M0a * f * exp(-t[i] / T1app) / R) * term;
      }

      // Arterial compartment (Eq. 2)
      if (t[i] >= att_a && t[i] <= att_a + tau) {
        deltaM[i] += 2 * a * M0a * abv * exp(-t[i] / T1a);
      }
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

  // Parameter values 
  real<lower=0> T1a_fixed;
  real<lower=0> T1_fixed;
  real<lower=0> lambd_fixed;
  real<lower=0> a_fixed;
  real<lower=0> abv_fixed;
  real<lower=0> att_a_fixed;

  // Fitting flags
  int<lower=0,upper=1> fit_T1a;
  int<lower=0,upper=1> fit_T1;
  int<lower=0,upper=1> fit_lambd;
  int<lower=0,upper=1> fit_abv;
  int<lower=0,upper=1> fit_att_a;

  // Priors 
  real T1a_prior_mean;
  real T1a_prior_std;
  real T1_prior_mean;
  real T1_prior_std;
  real lambd_prior_mean;
  real lambd_prior_std;
  real abv_prior_mean;
  real abv_prior_std;
  real att_a_prior_mean;
  real att_a_prior_std;

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
  real abv_lower;
  real abv_upper;
  real att_a_lower;
  real att_a_upper;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=0.001, upper=0.2> f;  // CBF as f parameter
  real<lower=0.001> sigma;

  // Conditional parameters
  real<lower=T1a_lower, upper=T1a_upper> T1a_param;
  real<lower=T1_lower, upper=T1_upper> T1_param;
  real<lower=lambd_lower, upper=lambd_upper> lambd_param;
  real<lower=abv_lower, upper=abv_upper> abv_param;
  real<lower=att_a_lower, upper=att_a_upper> att_a_param;
}

transformed parameters {
  real T1a_use;
  real T1_use;
  real lambd_use;
  real abv_use;
  real att_a_use;
  real cbf;  // CBF in ml/min/100g

  // Use fitted or constant fixed values
  T1a_use = fit_T1a ? T1a_param : T1a_fixed;
  T1_use = fit_T1 ? T1_param : T1_fixed;
  lambd_use = fit_lambd ? lambd_param : lambd_fixed;
  abv_use = fit_abv ? abv_param : abv_fixed;
  att_a_use = fit_att_a ? att_a_param : att_a_fixed;

  // Convert f to CBF
  cbf = f * 6000.0;
}

model {
  vector[n] mu;

  // Priors for ATT and CBF
  if (use_att_prior_from_ls == 1) {
    att ~ normal(att_prior_from_ls, att_prior_std);
  } else {
    att ~ normal(1.2, 0.5);
  }

  if (use_cbf_prior_from_ls == 1) {
    f ~ normal(cbf_prior_from_ls / 6000.0, cbf_prior_std / 6000.0);
  } else {
    f ~ normal(0.01, 0.0025);  // Default: 60 ml/min/100g
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
  if (fit_abv) {
    abv_param ~ normal(abv_prior_mean, abv_prior_std);
  }
  if (fit_att_a) {
    att_a_param ~ normal(att_a_prior_mean, att_a_prior_std);
  }

  // Likelihood
  mu = deltaM_model_ext(t, att, f, M0a, tau, abv_use, att_a_use, 
                        T1a_use, T1_use, lambd_use, a_fixed);
  signal ~ normal(mu, sigma);
}