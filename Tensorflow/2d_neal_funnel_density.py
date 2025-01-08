import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


# Define the potential energy function based on the specified density
def potential_energy_custom(coords):
    # Ensure q1 and q2 are treated as float64
    q1 = tf.cast(coords[0], tf.float64)
    q2 = tf.cast(coords[1], tf.float64)
    
    # Compute the negative log probability for q1 and q2
    log_prob_q1 = -tf.math.log(3 * np.sqrt(2 * np.pi)) - (q1**2 / (2 * 3))
    log_prob_q2 = -tf.math.log(tf.exp(q1) * np.sqrt(2 * np.pi)) - (q2**2 / (2 * tf.exp(q1)**2))
    
    return -(log_prob_q1 + log_prob_q2)

# The rest of your code would follow here...
# Log probability function for NUTS
def target_log_prob_fn(coords):
    return -potential_energy_custom(coords)

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
    return np.random.normal(size=(num_samples, 2))

# Parameters
num_samples = 25000
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
plt.title('Scatter Plot of Samples from NUTS and L-HNN')
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
plt.title('eCDF Comparison for Dimension q1')
plt.xlabel('q1')
plt.ylabel('eCDF')
plt.legend()
plt.grid()

# Plotting eCDFs for q2
plt.figure(figsize=(12, 6))
plt.plot(q2_nuts, ecdf_nuts_q2, label='NUTS q2', color='blue')
plt.plot(q2_l_hnn, ecdf_l_hnn_q2, label='L-HNN q2', color='orange')
plt.title('eCDF Comparison for Dimension q2')
plt.xlabel('q2')
plt.ylabel('eCDF')
plt.legend()
plt.grid()

plt.show()
