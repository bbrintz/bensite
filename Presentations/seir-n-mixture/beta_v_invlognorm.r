library(tidyverse)
library(patchwork)

# Set parameters
#out= 1:1000 %>% purrr::map(function(x){
n <- 2000000
p <- 0.1
nrep <- 1e4

# Ground truth binomial simulation 
binom_sim <- rbinom(nrep, size = n, prob = p)/n



# Approach 1: Beta approximation for the proportion
# For a binom(n, p), a Beta( n*p, n*(1-p) ) gives a similar mean/variance for X/n.
# Then we scale by n and round.
beta_sim <-  rbeta(nrep, shape1 = n * p, shape2 = n * (1-p))

# Approach 2: Inverse normal logit approximation
# Using the delta method, the logit (i.e., log(p/(1-p))) will have variance approx 1/(n*p*(1-p))
logit_mean <- log(p / (1 - p))
logit_sd <- sqrt(1 / (n * p * (1-p)))  # delta method variance

invlogit_sim <- plogis(rnorm(nrep, mean = logit_mean, sd = logit_sd)) 


# Create a data frame for comparison
df <- tibble(
  outcome = c(binom_sim, binom_sim, beta_sim, invlogit_sim),
  plt = rep(c("plt1","plt2","plt1","plt2"), each = nrep),
  method = rep(c("Binom", "Binom","Beta Approx", "InvLogit Approx"), each = nrep)
)

# Plot histograms of the three approaches:
df %>%#filter(method!= "Beta Approx") %>%
  ggplot(aes(x = outcome, fill = method)) +
  geom_density(alpha=.25) +
  theme_minimal() + facet_wrap(~plt) + scale_fill_viridis_d(begin=.25,end=.75)



#as.numeric(ks.test(binom_sim, beta_sim)$p<ks.test(binom_sim, invlogit_sim)$p)})


#mean(unlist(out))
