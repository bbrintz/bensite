library(tidyverse)
library(gganimate)
library(patchwork)

# Parameters
n <- 2000000
p <- 0.1
nrep <- 1e4

# Set up parameters for the inverse logit approach
logit_mean <- log(p / (1 - p))
logit_sd <- sqrt(1 / (n * p * (1 - p)))

# Simulate samples
set.seed(123)
normal_samples <- rnorm(nrep, mean = logit_mean, sd = logit_sd)
invlogit_samples <- plogis(normal_samples)
binom_sim <- rbinom(nrep, size = n, prob = p)/n

# Create a frame index so that we can build the densities gradually
df_normal <- tibble(
  sample = normal_samples,
  frame = 1:nrep
)
df_invlogit <- tibble(
  sample = invlogit_samples,
  frame = 1:nrep
)
df_binom <- tibble(
  sample = binom_sim,
  frame = 1:nrep
)

# Left panel: density of normal samples (logit scale)
p1 <- ggplot(df_normal, aes(x = sample)) +
  geom_density(fill = "cornflowerblue", alpha = 0.5) +
  labs(title = "Cumulative Density of Normal Samples (Logit Scale)",
       x = "Logit Value", y = "Density") +
  transition_time(frame) +
  ease_aes('linear')

# Right panel: density of inverse logit samples versus binomial density
p2 <- ggplot() +
  geom_density(data = df_invlogit, aes(x = sample), fill = "orange", alpha = 0.5) +
  geom_density(data = df_binom, aes(x = sample), color = "black", size = 1, linetype = "dashed") +
  labs(title = "Cumulative Density: Inverse Logit vs. Binom", 
       x = "Proportion", y = "Density") +
  transition_time(frame) +
  ease_aes('linear')

# Combine the two panels using patchwork:
combined <- p1 + p2 + plot_layout(ncol = 2, widths = c(1, 1))

# Animate
anim <- animate(combined, nframes = 200, fps = 20, width = 900, height = 400)
anim
