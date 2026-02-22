data {
  int<lower=1> T;
  array[T] int<lower=0> y;
  array[T] int<lower=1, upper=7> dow; // 1=Mon,...,7=Sun (or your mapping)
  // annual seasonality via Fourier harmonics
  int<lower=1> P;                 // seasonal period in time units (e.g., 365 for daily data)
  int<lower=0> K;                 // number of Fourier harmonics to include (K=0 disables seasonality)
  real<lower=0> season_decay;     // decay exponent for harmonic scales (e.g., 1.5)
  int<lower=0> J;                 // number of logistic ramps (0 allows none)
}

transformed data {
  // Precompute Fourier bases to avoid K*T trig calls each leap in the leapfrog
  matrix[T, K] C;  // cos(2π k t / P)
  matrix[T, K] S;  // sin(2π k t / P)
  for (t in 1:T) {
    for (k in 1:K) {
      real omega = 2 * pi() * k / P;
      C[t, k] = cos(omega * t);
      S[t, k] = sin(omega * t);
    }
  }
}


parameters {
  // multiple smooth change-points (logistic ramps)
  real m0;                 // baseline log-rate level
  vector[J] d_raw;         // raw step sizes for horseshoe
  vector<lower=0>[J] lambda;   // local shrinkage scales
  real<lower=0> hs_tau;        // global shrinkage scale
  vector[J] tau_unconstrained;  // any real values
  vector[J] log_s;         // log widths for each transition (s_j = exp(log_s[j]))

  // NB dispersion (regularize distance from Poisson)
  real<lower=0> inv_phi;  // = 1/phi

  // weekday effects with sum-to-zero via last constructed
  vector[6] gamma_raw;    // gamma[1:6] free; gamma[7] = -sum(gamma[1:6])
  real<lower=0> sigma_gamma;

  // hurdle weekday effects with sum-to-zero via last constructed
  vector[6] eta_raw;      // eta[1:6] free; eta[7] = -sum(eta[1:6])
  real<lower=0> sigma_eta;

  // heavy-tailed daily shock for spikes/anomalies on the log-rate
  vector[T] kappa_raw;    // non-centered shock term (standard normal)
  real<lower=0> sigma_kappa; // global scale for shocks

  // AR(1) residual on log-rate (non-centered)
  real<lower=-1, upper=1> rho;
  real<lower=0> sigma_z;
  vector[T] eps_z;        // standard-normal innovations


  vector[K] a_cos;  // cos coefficients
  vector[K] b_sin;  // sin coefficients
  real<lower=0> sigma_season; // global scale for seasonality

  // hurdle (probability of any event)
  real h0;                 // global intercept for zero-inflation probability

}

transformed parameters {
    
  vector[J] tau;
  vector[J] s;
  vector[J] d;             // horseshoe-shrunk step sizes
  ordered[J] tau_raw = sort_asc(inv_logit(tau_unconstrained));

  for (j in 1:J) {
    tau[j] = 1 + (T - 1) * tau_raw[j];
    s[j]   = exp(log_s[j]);
  }
  d = d_raw .* lambda * hs_tau;

  real phi = 1 / inv_phi;

  // weekday effects, sum-to-zero
  vector[7] gamma;
  gamma[1:6] = gamma_raw;
  gamma[7]   = -sum(gamma_raw);

  // hurdle weekday effects, sum-to-zero
  vector[7] eta;
  eta[1:6] = eta_raw;
  eta[7]   = -sum(eta_raw);

  // AR(1) state (non-centered)
  vector[T] z;
  z[1] = eps_z[1] * (sigma_z / sqrt(1 - square(rho)));   // stationary prior sd
  for (t in 2:T) z[t] = rho * z[t-1] + sigma_z * eps_z[t];

  // non-centered kappa
  vector[T] kappa;
  kappa = sigma_kappa * kappa_raw;


  // seasonal component via K Fourier harmonics with period P
  vector[T] season;
  season = rep_vector(0, T);
  if (K > 0) {
    season = C * a_cos + S * b_sin; // O(T*K) but without trig in the inner loop
  }
  
  vector[T] logit_zi;
  for (t in 1:T) {
    logit_zi[t] = h0 + eta[dow[t]];  // zero-inflation logit (structural zero probability)
  }

  // construct log-lambda_t with J smooth logistic transitions
  vector[T] log_lambda;
  for (t in 1:T) {
    real ramps = m0; // baseline level
    if (J > 0) {
      for (j in 1:J) {
        // smooth step that goes from ~0 (well before tau[j]) to ~1 (well after tau[j])
        ramps += d[j] * inv_logit( (t - tau[j]) / s[j] );
      }
    }
    log_lambda[t] = ramps + gamma[dow[t]] + z[t] + kappa[t] + season[t];
  }
  real diff = exp(m0 + sum(d)) - exp(m0); // total change in the lambda scale from start to far after last change
}

model {
  // priors on multiple change component
  m0      ~ normal(0, 1);
  // horseshoe prior for step sizes
  d_raw   ~ normal(0, 1);
  lambda  ~ cauchy(0, 1);
  hs_tau  ~ normal(0, 0.3);   // strong global shrinkage toward no change
  tau_unconstrained ~ normal(0, 1); // uniform over (1, T) after scaling, ordered
  log_s   ~ normal(0.5, 0.5);             // per-change widths (typical span ~ exp(0.5) ≈ 1.65 time units)
 
  h0 ~ normal(-1, 1.5);   // weakly favors low usage but not extreme


  // NB overdispersion (half-normal on 1/phi)
  inv_phi ~ normal(0, 0.3);

  // weekday effects
  sigma_gamma ~ normal(0, 0.3);
  gamma_raw   ~ normal(0, sigma_gamma);

  // hurdle weekday effects
  sigma_eta ~ normal(0, 0.3);
  eta_raw   ~ normal(0, sigma_eta);

  // shocks: heavy-tailed to allow occasional large spikes while shrinking most days
  sigma_kappa ~ normal(0, 0.2);                 // encourages small typical shocks
  kappa_raw   ~ normal(0, 1);                   // non-centered shocks

  // AR(1) components (non-centered)
  sigma_z ~ normal(0, 0.3);
  rho     ~ normal(0, 0.5);
  eps_z   ~ normal(0, 1);


  // seasonality priors (shrink higher harmonics more strongly)
  if (K > 0) {
    vector[K] sd_k;
    for (k in 1:K) sd_k[k] = sigma_season / pow(k, season_decay); // e.g., decay ~ 1.5
    sigma_season ~ normal(0, 0.3);
    a_cos ~ normal(0, sd_k);
    b_sin ~ normal(0, sd_k);
  }


  // weighted likelihood (power-likelihood). set w[t]=1 to match original model

for (t in 1:T) {
  if (y[t] == 0) {
    // log( pi + (1-pi) * NB(0) )
    target += log_mix(inv_logit(logit_zi[t]),
                      0,
                      neg_binomial_2_log_lpmf(0 | log_lambda[t], phi));
  } else {
    // log( (1-pi) * NB(y) )
    target += bernoulli_logit_lpmf(0 | logit_zi[t])
              + neg_binomial_2_log_lpmf(y[t] | log_lambda[t], phi);
  }
}
}

generated quantities {
  array[T] int y_rep;
  array[T] real log_lik;

  for (t in 1:T) {
    // pointwise log-likelihood under ZINB
    if (y[t] == 0) {
      log_lik[t] = log_mix(inv_logit(logit_zi[t]),
                           0,
                           neg_binomial_2_log_lpmf(0 | log_lambda[t], phi));
    } else {
      log_lik[t] = bernoulli_logit_lpmf(0 | logit_zi[t])
                   + neg_binomial_2_log_lpmf(y[t] | log_lambda[t], phi);
    }

    // posterior predictive draw under ZINB
    if (bernoulli_rng(inv_logit(logit_zi[t])) == 1) {
      y_rep[t] = 0;
    } else {
      y_rep[t] = neg_binomial_2_log_rng(log_lambda[t], phi);
    }
  }
}
