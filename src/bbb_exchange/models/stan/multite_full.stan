// multi_te_full.stan
data {
  // Measurements
  int<lower=1> n_measurements;        // total number of TE points over all TI
  int<lower=1> n_ti;                  // number different TI
  array[n_measurements] real signal;  // measured DeltaM (normalised)

  // Timing
  array[n_ti] real tis;               // TI[j]
  array[n_ti] int<lower=1> ntes;      // number TE for each TI
  array[n_measurements] real tes;    
  array[n_ti] real tau_per_ti;        // labelling time

  real<lower=0> t1;
  real<lower=0> t1b;
  real<lower=0> t2;
  real<lower=0> t2b;
  real<lower=0> texch;
  real<lower=0> itt;
  real<lower=0> lambd;
  real<lower=0> alpha;
  real<lower=0> M0a;

  // Priors
  real att_prior_mean;
  real<lower=0> att_prior_std;
  real cbf_prior_mean;
  real<lower=0> cbf_prior_std;
}

parameters {
  real<lower=0.1, upper=3.0> att;       // s
  real<lower=10.0, upper=250.0> cbf;    // ml/min/100g
  real<lower=1e-6> sigma;               // noise
}

transformed parameters {
  // conversion: cbf [ml/min/100g] -> f [ml/s/g]
  real f = (cbf / 100.0) * 60.0 / 6000.0;
  vector[n_measurements] mu;

  {
    int te_index = 1; // Stan-index starts at 1 (not 0)
    vector[n_measurements] S_bl1_final = rep_vector(0.0, n_measurements);
    vector[n_measurements] S_bl2_final = rep_vector(0.0, n_measurements);
    vector[n_measurements] S_ex_final  = rep_vector(0.0, n_measurements);

    for (j in 1:n_ti) {
      real tau = tau_per_ti[j];
      real ti  = tis[j];

      // === Case 1: 0 < ti < att ===
      if ((0 < ti) && (ti < att)) {
        for (k in 1:ntes[j]) {
          S_bl1_final[te_index] = 0;
          S_bl2_final[te_index] = 0;
          S_ex_final[te_index]  = 0;
          te_index += 1;
        }
      }

      // === Case 2: att <= ti < (att + itt) ===
      else if ((att <= ti) && (ti < att + itt)) {
        for (k in 1:ntes[j]) {
          real te = tes[te_index];
          if ((0 <= te) && (te < (att + itt - ti))) {
            S_bl1_final[te_index] =
              (2 * f * t1b * exp(-att / t1b) * exp(-ti / t1b)
               * (exp(ti / t1b) - exp(att / t1b)) * exp(-te / t2b));
          }
          else if (((att + itt - ti) <= te) && (te < itt)) {
            real base_term = 2 * f * t1b * exp(-att / t1b) * exp(-ti / t1b)
                             * (exp(ti / t1b) - exp(att / t1b));
            real transition_factor = (te - (att + itt - ti)) / (ti - att);
            S_bl1_final[te_index] = (base_term - transition_factor * base_term) * exp(-te / t2b);
            S_bl2_final[te_index] = (transition_factor * base_term) * exp(-te / t2b) * exp(-te / texch);
            S_ex_final[te_index]  = (transition_factor * base_term) * (1 - exp(-te / texch)) * exp(-te / t2);
          }
          else {
            S_bl2_final[te_index] =
              (2 * f * t1b * exp(-att / t1b) * exp(-ti / t1b)
               * (exp(ti / t1b) - exp(att / t1b)) * exp(-te / t2b) * exp(-te / texch));
            S_ex_final[te_index]  =
              (2 * f * t1b * exp(-att / t1b) * exp(-ti / t1b)
               * (exp(ti / t1b) - exp(att / t1b)) * (1 - exp(-te / texch)) * exp(-te / t2));
          }
          te_index += 1;
        }
      }

      // === Case 3: (att+itt) <= ti < (att + tau) ===
      else if (((att + itt) <= ti) && (ti < (att + tau))) {
        for (k in 1:ntes[j]) {
          real te = tes[te_index];
          real term1 = 2 * f * t1b * exp(-att / t1b) * exp(-ti / t1b)
                       * (exp(ti / t1b) - exp(att / t1b));
          real term2 = 2 * f * t1b * exp(-(att + itt) / t1b) * exp(-ti / t1b)
                       * (exp(ti / t1b) - exp((att + itt) / t1b));
          real base_diff = term1 - term2;
          if ((0 <= te) && (te < itt)) {
            real transition_factor = te / itt;
            S_bl1_final[te_index] = (base_diff - transition_factor * base_diff) * exp(-te / t2b);
            S_bl2_final[te_index] =
              ((2 * f * exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch))
                * exp(-((1 / t1b) + (1 / texch)) * ti)
                * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
               + transition_factor * base_diff) * exp(-te / t2b) * exp(-te / texch));
            S_ex_final[te_index] =
              (((2 * f * exp(-(1 / t1b) * (att + itt))) / (1 / t1) * exp(-(1 / t1) * ti)
                  * (exp((1 / t1) * ti) - exp((1 / t1) * (att + itt)))
               - (2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1))
                  * exp(-((1 / t1) + (1 / texch)) * ti)
                  * (exp(((1 / texch) + (1 / t1)) * ti) - exp(((1 / texch) + (1 / t1)) * (att + itt))))
               * exp(-te / t2))
              + (((2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / t1b) + (1 / texch))
                   * exp(-((1 / t1b) + (1 / texch)) * ti)
                   * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
                 + base_diff) * (1 - exp(-te / texch)) * exp(-te / t2));
          } else {
            S_bl2_final[te_index] =
              ((2 * f * exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch))
                * exp(-((1 / t1b) + (1 / texch)) * ti)
                * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
               + base_diff) * exp(-te / t2b) * exp(-te / texch));
            S_ex_final[te_index] =
              (((2 * f * exp(-(1 / t1b) * (att + itt))) / (1 / t1) * exp(-(1 / t1) * ti)
                  * (exp((1 / t1) * ti) - exp((1 / t1) * (att + itt)))
               - (2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1))
                  * exp(-((1 / t1) + (1 / texch)) * ti)
                  * (exp(((1 / texch) + (1 / t1)) * ti) - exp(((1 / texch) + (1 / t1)) * (att + itt))))
               * exp(-te / t2))
              + (((2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / t1b) + (1 / texch))
                   * exp(-((1 / t1b) + (1 / texch)) * ti)
                   * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
                 + base_diff) * (1 - exp(-te / texch)) * exp(-te / t2));
          }
          te_index += 1;
        }
      }

      // === Case 4: (att + tau) <= ti < (att + itt + tau) ===
      else if (((att + tau) <= ti) && (ti < (att + itt + tau))) {
        for (k in 1:ntes[j]) {
          real te = tes[te_index];
          real term1 = 2 * f * t1b * exp(-att / t1b) * exp(-ti / t1b)
                       * (exp((att + tau) / t1b) - exp(att / t1b));
          real term2 = 2 * f * t1b * exp(-(att + itt) / t1b) * exp(-ti / t1b)
                       * (exp(ti / t1b) - exp((att + itt) / t1b));
          real base_diff = term1 - term2;
          if ((0 <= te) && (te < (itt - (ti - (att + tau))))) {
            real transition_factor = te / (itt - (ti - (att + tau)));
            S_bl1_final[te_index] = (base_diff - transition_factor * base_diff) * exp(-te / t2b);
            S_bl2_final[te_index] =
              ((2 * f * exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch))
                * exp(-((1 / t1b) + (1 / texch)) * ti)
                * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
               + transition_factor * base_diff) * exp(-te / t2b) * exp(-te / texch));
            S_ex_final[te_index] =
              (((2 * f * exp(-(1 / t1b) * (att + itt))) / (1 / t1) * exp(-(1 / t1) * ti)
                  * (exp((1 / t1) * ti) - exp((1 / t1) * (att + itt)))
               - (2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1))
                  * exp(-((1 / t1) + (1 / texch)) * ti)
                  * (exp(((1 / texch) + (1 / t1)) * ti) - exp(((1 / texch) + (1 / t1)) * (att + itt))))
               * exp(-te / t2))
              + (((2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / t1b) + (1 / texch))
                   * exp(-((1 / t1b) + (1 / texch)) * ti)
                   * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
                 + transition_factor * base_diff) * (1 - exp(-te / texch)) * exp(-te / t2));
          } else {
            S_bl2_final[te_index] =
              ((2 * f * exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch))
                * exp(-((1 / t1b) + (1 / texch)) * ti)
                * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
               + base_diff) * exp(-te / t2b) * exp(-te / texch));
            S_ex_final[te_index] =
              (((2 * f * exp(-(1 / t1b) * (att + itt))) / (1 / t1) * exp(-(1 / t1) * ti)
                  * (exp((1 / t1) * ti) - exp((1 / t1) * (att + itt)))
               - (2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1))
                  * exp(-((1 / t1) + (1 / texch)) * ti)
                  * (exp(((1 / texch) + (1 / t1)) * ti) - exp(((1 / texch) + (1 / t1)) * (att + itt))))
               * exp(-te / t2))
              + (((2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / t1b) + (1 / texch))
                   * exp(-((1 / t1b) + (1 / texch)) * ti)
                   * (exp(((1 / t1b) + (1 / texch)) * ti) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
                 + base_diff) * (1 - exp(-te / texch)) * exp(-te / t2));
          }
          te_index += 1;
        }
      }

      // === Case 5: ti >= (att + itt + tau) ===
      else {
        for (k in 1:ntes[j]) {
          real te = tes[te_index];
          S_bl2_final[te_index] =
            (2 * f * exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch))
             * exp(-((1 / t1b) + (1 / texch)) * ti)
             * (exp(((1 / t1b) + (1 / texch)) * (att + itt + tau)) - exp(((1 / t1b) + (1 / texch)) * (att + itt)))
             * exp(-te / t2b) * exp(-te / texch));
          S_ex_final[te_index] =
            (((2 * f * exp(-(1 / t1b) * (att + itt))) / (1 / t1) * exp(-(1 / t1) * ti)
                * (exp((1 / t1) * (att + itt + tau)) - exp((1 / t1) * (att + itt)))
             - (2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1))
                * exp(-((1 / t1) + (1 / texch)) * ti)
                * (exp(((1 / texch) + (1 / t1)) * ti) - exp(((1 / texch) + (1 / t1)) * (att + itt))))
             * exp(-te / t2))
            + (((2 * f * exp(-(1 / t1b) * (att + itt))) / ((1 / t1b) + (1 / texch))
                 * exp(-((1 / t1b) + (1 / texch)) * ti)
                 * (exp(((1 / t1b) + (1 / texch)) * (att + itt + tau)) - exp(((1 / t1b) + (1 / texch)) * (att + itt))))
               * (1 - exp(-te / texch)) * exp(-te / t2));
          te_index += 1;
        }
      }
    }

    mu = (S_bl1_final + S_bl2_final + S_ex_final) * (M0a * alpha / lambd);
  }
}

model {
  // Priors
  att  ~ normal(att_prior_mean, att_prior_std);
  cbf  ~ normal(cbf_prior_mean, cbf_prior_std);
  sigma ~ exponential(1.0);

  // Likelihood
  signal ~ normal(mu, sigma);
}

generated quantities {
  vector[n_measurements] mu_ppc;
  for (i in 1:n_measurements) {
    mu_ppc[i] = normal_rng(mu[i], sigma);
  }
}