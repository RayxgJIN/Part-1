import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Define the target distribution (e.g., a normal distribution)
def target_log_prob_fn(x):
    return tfp.distributions.Normal(loc=0., scale=1.).log_prob(x)

# Set up HMC parameters
step_size = 0.1
num_leapfrog_steps = 3

# Create HMC kernel
hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=step_size,
    num_leapfrog_steps=num_leapfrog_steps)

# Initialize the chain
initial_position = tf.constant(0.0)

# Run the HMC sampling
num_samples = 1000
samples, is_accepted = tfp.mcmc.sample_chain(
    num_results=num_samples,
    num_burnin_steps=500,
    current_state=initial_position,
    kernel=hmc_kernel)

# Convert samples to numpy for analysis
samples_numpy = samples.numpy()

# Plot the histogram of the samples
plt.figure(figsize=(10, 6))
plt.hist(samples_numpy, bins=30, density=True, alpha=0.6, color='b')

# Overlay the target distribution for comparison
x = tf.linspace(-4.0, 4.0, 100)
plt.plot(x.numpy(), tfp.distributions.Normal(loc=0., scale=1.).prob(x).numpy(), 'r', lw=2)

plt.title('HMC Sampling from a Normal Distribution')
plt.xlabel('Sample Value')
plt.ylabel('Density')
plt.grid()
plt.show()