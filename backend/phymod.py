
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
# Set default data type to float32
tf.keras.backend.set_floatx('float32')

# Step 1: Load and Prepare Data
# ------------------------------------------------
# Load dataset (replace 'distillation_data.csv' with your file)
data = pd.read_csv('converted_distillation_data.csv')
print(data.columns)
# Define features (input) and targets (output)
X = data[['Feed_Flow_Rate', 'Feed_Composition', 'Reflux_Ratio', 'Boil_Up_Ratio']]
y = data[['Distillate_Purity', 'Reboiler_Duty']]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (important for neural networks)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train).astype(np.float32)
X_test = scaler_X.transform(X_test).astype(np.float32)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train).astype(np.float32)
y_test = scaler_y.transform(y_test).astype(np.float32)

# Step 2: Physics-Based Model
# ------------------------------------------------
def physics_model(X):
    """
    Implements a simple physics-based model for the distillation column.
    Input: X (normalized features)
    Output: y_physics (normalized predictions based on physics)
    """
    # Denormalize input features
    X_denorm = scaler_X.inverse_transform(X)
    F = X_denorm[:, 0]  # Feed flow rate
    z = X_denorm[:, 1]  # Feed composition
    R = X_denorm[:, 2]  # Reflux ratio
    B = X_denorm[:, 3]  # Boil-up ratio

    # Physics-based calculations (simplified)
    D = F * 0.5  # Distillate flow rate (example)
    x_D = z * 0.9  # Distillate purity (example)
    Q_R = F * 100  # Reboiler duty (example)

    # Combine outputs
    y_physics = np.column_stack([x_D, Q_R])

    # Normalize outputs
    y_physics_norm = scaler_y.transform(y_physics).astype(np.float32)
    return y_physics_norm

# Step 3: Hybrid Model (Physics-Informed Neural Network)
# ------------------------------------------------
# Define the physics-based loss function
def physics_loss(y_true, y_pred):
    """
    Combines data-driven loss with physics-based constraints.
    """
    # Ensure y_true and y_pred are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Data-driven loss (MSE)
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Physics-based loss (mass balance example)
    X_denorm = scaler_X.inverse_transform(X_train)  # Denormalize inputs
    F = X_denorm[:, 0]  # Feed flow rate
    D = F * 0.5  # Distillate flow rate (example)
    B = F - D  # Bottoms flow rate (mass balance)
    mass_balance_error = tf.reduce_mean(tf.square(F - (D + B)))

    # Combine losses
    total_loss = data_loss + 0.1 * mass_balance_error  # Weight for physics loss
    return total_loss

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer
    tf.keras.layers.Dense(y_train.shape[1])  # Output layer
])

# Compile the model with physics-based loss
model.compile(optimizer='adam', loss=physics_loss)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Step 4: Evaluate the Model
# ------------------------------------------------
# Evaluate on test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

# Make predictions
y_pred = model.predict(X_test)

# Denormalize predictions and true values
y_pred_denorm = scaler_y.inverse_transform(y_pred)
y_test_denorm = scaler_y.inverse_transform(y_test)

# Step 5: Visualize Results
# ------------------------------------------------
# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.show()

# Scatter plot for true vs predicted values
plt.scatter(y_test_denorm[:, 0], y_pred_denorm[:, 0], label='Distillate Purity')
plt.scatter(y_test_denorm[:, 1], y_pred_denorm[:, 1], label='Reboiler Duty')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.title('True vs Predicted')
plt.show()

# Step 6: Compare with Pure Physics-Based Model
# ------------------------------------------------
# Generate physics-based predictions
y_physics = physics_model(X_test)
y_physics_denorm = scaler_y.inverse_transform(y_physics)

# Debugging: Check shapes and data
print("Shape of y_test_denorm:", y_test_denorm.shape)
print("Shape of y_pred_denorm:", y_pred_denorm.shape)
print("Shape of y_physics_denorm:", y_physics_denorm.shape)

print("y_test_denorm:", y_test_denorm)
print("y_pred_denorm:", y_pred_denorm)
print("y_physics_denorm:", y_physics_denorm)

# Compare RMSE for ML model and physics-based model
try:
    ml_rmse = root_mean_squared_error(y_test_denorm, y_pred_denorm)
    physics_rmse = root_mean_squared_error(y_test_denorm, y_physics_denorm)
    print("ML Model RMSE:", ml_rmse)
    print("Physics Model RMSE:", physics_rmse)
except Exception as e:
    print("Error calculating RMSE:", e)
