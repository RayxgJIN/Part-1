import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Define Neal's funnel potential energy function
def potential_energy_neal_funnel(coords):
    q1, q2 = coords[0], coords[1]
    return 0.5 * (q1**2 / 9) + 0.5 * (q2**2) * tf.exp(q1 / 3)

# Log probability function for NUTS
def target_log_prob_fn(coords):
    return -potential_energy_neal_funnel(coords)

# NUTS Sampling
def sample_nuts(initial_position, num_samples, step_size=0.1):
    nuts_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size
    )
    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=initial_position,
        kernel=nuts_kernel
    )
    return samples.numpy()

# Simulate L-HNN Sampling (for demonstration)
def sample_l_hnn(initial_position, num_samples):
    # Placeholder for L-HNN sampling logic
    # Here, we just generate random samples from a normal distribution for demonstration
    return np.random.normal(size=(num_samples, 2))

# Parameters
num_samples = 1000
initial_position = tf.constant([0.0, 0.0], dtype=tf.float32)

# Generate samples
nuts_samples = sample_nuts(initial_position, num_samples)
l_hnn_samples = sample_l_hnn(initial_position, num_samples)

# Plotting the samples
plt.figure(figsize=(12, 6))

# Scatter plot comparison
plt.subplot(1, 2, 1)
plt.scatter(nuts_samples[:, 0], nuts_samples[:, 1], alpha=0.5, label='NUTS', color='blue')
plt.scatter(l_hnn_samples[:, 0], l_hnn_samples[:, 1], alpha=0.5, label='L-HNN', color='orange')
plt.title('Scatter Plot of Samples')
plt.xlabel('q1')
plt.ylabel('q2')
plt.legend()
plt.axis('equal')
plt.grid()

# eCDF Calculation
def empirical_cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, yvals

# eCDF for q1 and q2
q1_nuts, ecdf_nuts_q1 = empirical_cdf(nuts_samples[:, 0])
q1_l_hnn, ecdf_l_hnn_q1 = empirical_cdf(l_hnn_samples[:, 0])
q2_nuts, ecdf_nuts_q2 = empirical_cdf(nuts_samples[:, 1])
q2_l_hnn, ecdf_l_hnn_q2 = empirical_cdf(l_hnn_samples[:, 1])

# Plotting eCDFs for q1
plt.subplot(1, 2, 2)
plt.plot(q1_nuts, ecdf_nuts_q1, label='NUTS q1', color='blue')
plt.plot(q1_l_hnn, ecdf_l_hnn_q1, label='L-HNN q1', color='orange')
plt.title('eCDF Comparison for q1')
plt.xlabel('q1')
plt.ylabel('eCDF')
plt.legend()
plt.grid()

# Plotting eCDFs for q2
plt.figure(figsize=(12, 6))
plt.plot(q2_nuts, ecdf_nuts_q2, label='NUTS q2', color='blue')
plt.plot(q2_l_hnn, ecdf_l_hnn_q2, label='L-HNN q2', color='orange')
plt.title('eCDF Comparison for q2')
plt.xlabel('q2')
plt.ylabel('eCDF')
plt.legend()
plt.grid()

plt.show()