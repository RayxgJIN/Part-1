import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Define the Gaussian mixture density function
def target_density(q):
    return 0.5 * tf.exp(-0.5 * ((q - 1) / 0.35)**2) + 0.5 * tf.exp(-0.5 * ((q + 1) / 0.35)**2)

# Hamiltonian dynamics simulation using the leapfrog method
def simulate_hamiltonian_dynamics(num_samples, T, num_steps, dt):
    positions = np.zeros((num_samples, T))
    momenta = np.zeros((num_samples, T))

    for i in range(num_samples):
        momenta[i, 0] = np.random.normal()  # Initial momentum drawn from a Gaussian
        positions[i, 0] = np.clip(0 if i == 0 else positions[i - 1, -1], -2, 2)  # Clip initial position

        for t in range(1, T):
            p = momenta[i, t - 1]
            q = positions[i, t - 1]

            # Update momentum (negative gradient)
            p -= dt * target_density(q).numpy()  # Gradient of potential energy
            # Update position
            q += dt * p
            
            # Clip position and momentum to keep them in range
            positions[i, t] = np.clip(q, -2, 2)
            momenta[i, t] = np.clip(p, -2, 2)

    return positions, momenta

# Traditional HMC sampling
def traditional_hmc(num_samples, num_steps, dt):
    samples = []
    total_gradients = 0  # Initialize gradient counter
    for _ in range(num_samples):
        q = np.random.uniform(-2, 2)  # Start uniformly within [-2, 2]
        p = np.random.normal()
        for _ in range(num_steps):
            # Calculate gradient for the current position
            gradient = target_density(q).numpy()
            p -= dt * gradient  # Update momentum
            q += dt * p  # Update position
            
            # Clip position to keep it in range
            q = np.clip(q, -2, 2)
            total_gradients += 1  # Count this gradient calculation
            
        samples.append(q)
    
    print(f'Total gradients calculated in traditional HMC: {total_gradients}')
    return np.array(samples)

# Define the L-HNN model
class L_HNN(tf.keras.Model):
    def __init__(self):
        super(L_HNN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(100, activation='relu') for _ in range(3)]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def build(self,input_shape):
        self.hidden_layers[0].build(input_shape)
        input_shape = self.hidden_layers[0].compute_output_shape(input_shape)
        for layer in self.hidden_layers[1:]:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.output_layer.build(input_shape)
        self.built = True

   

# Training the L-HNN
def train_l_hnn(positions, epochs=400):
    model = L_HNN()
    model.build(input_shape=(None, 2))

    optimizer = tf.keras.optimizers.Adam()
    num_trainable_params = np.sum([np.prod(var.shape) for var in model.trainable_variables])
    print(f'Number of gradients to be calculated for L-HNN: {num_trainable_params}')

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            idx = np.random.choice(positions.shape[0], size=20)
            x = tf.convert_to_tensor(positions[idx, :], dtype=tf.float32)

            # Compute density values
            density_values = target_density(x)

            # Check for NaN or Inf in density values
            if tf.reduce_any(tf.math.is_nan(density_values)) or tf.reduce_any(tf.math.is_inf(density_values)):
                print(f"Invalid density values at epoch {epoch}: {density_values.numpy()}")
                continue
            
            # Simplified loss function for troubleshooting
            loss = -tf.reduce_mean(tf.where(density_values > 0, tf.math.log(density_values), tf.zeros_like(density_values)))
            #loss = -tf.reduce_mean(density_values)

            if tf.reduce_any(tf.math.is_nan(loss)) or tf.reduce_any(tf.math.is_inf(loss)):
                print(f"Invalid loss value at epoch {epoch}: {loss.numpy()}")
                continue  # Skip this iteration if loss is invalid

            grads = tape.gradient(loss, model.trainable_variables)

        # Check if gradients are None

        total_gradients = len(grads)
        print(f'Gradients calculated for L-HNN at epoch {epoch}: {total_gradients}')

        if grads is None or all(g is None for g in grads):
            print(f"No gradients computed at epoch {epoch}. Skipping update.")
            continue  # Skip update if no gradients are computed

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')
    
    return model
# Parameters
num_samples = 1000
T = 20
num_steps = 20
dt = 0.05

# Simulate Hamiltonian dynamics for L-HNN
try:
    positions, momenta = simulate_hamiltonian_dynamics(num_samples, T, num_steps, dt)
    print("Successfully unpacked positions and momenta.")
except Exception as e:
    print(f"Error in simulate_hamiltonian_dynamics: {e}")

# Train the L-HNN
l_hnn_model = train_l_hnn(positions)

# Sampling using traditional HMC
traditional_samples = traditional_hmc(1000, num_steps, dt)

# Sampling using L-HNN
l_hnn_samples = l_hnn_model(tf.convert_to_tensor(np.random.uniform(-2, 2, size=(1000, 2)), dtype=tf.float32))

# Phase space plot
plt.figure(figsize=(18, 6))
# Create a range of values for the Gaussian mixture density
q_values = np.linspace(-2, 2, 100)
density_values = target_density(tf.convert_to_tensor(q_values)).numpy()

# (a) Phase space plot
plt.subplot(1, 3, 1)
for energy_level in [0.1, 0.5, 1.0]:  # Example energy levels
    q = np.linspace(-2, 2, 100)  # Adjusted range
    target_values = target_density(q).numpy()
    # Ensure the expression is non-negative
    energy_expression = 2 * energy_level - 2 * target_values
    energy_expression = np.clip(energy_expression, 0, None)  # Clip to avoid negative values

    p = np.sqrt(energy_expression)
    plt.plot(q, p, label=f'E = {energy_level}')
    plt.plot(q, -p)

# (b) Histogram of samples from traditional HMC
plt.subplot(1, 3, 2)
plt.hist(traditional_samples, bins=30, density=False, alpha=0.7, label='Traditional HMC', color='blue')
plt.plot(q_values, density_values * num_samples * (3 / 30), color='red', label='Gaussian Mixture Density', linewidth=2)  # Scale density for comparison
plt.title('Histogram of Samples from Traditional HMC')
plt.xlabel('Sampled value')
plt.ylabel('Count')  # Change label to 'Count'
plt.legend()

# (c) Histogram of samples from L-HNN
plt.subplot(1, 3, 3)
plt.hist(l_hnn_samples.numpy(), bins=30, density=False, alpha=0.7, label='L-HNN in HMC', color='orange')
plt.plot(q_values, density_values * num_samples * (3 / 30), color='red', label='Gaussian Mixture Density', linewidth=2)  # Scale density for comparison
plt.title('Histogram of Samples from L-HNN in HMC')
plt.xlabel('Sampled value')
plt.ylabel('Count')  # Change label to 'Count'
plt.legend()

plt.tight_layout()

plt.savefig("result1.png",dpi=200)
plt.show()