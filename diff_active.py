import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# 加载数据
file_path = 'data/merged_all.xlsx'
data = pd.read_excel(file_path)

# 特征和标签
feature_columns = [col for col in data.columns if col != 'Pathogen Load']
X = data[feature_columns]
y = data['Pathogen Load']

# 对数据进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

def shifted_relu(x):
    return tf.maximum(-1.0, x)

def create_model(activation_function):
    input_x = tf.keras.layers.Input(shape=(144,))
    x = tf.keras.layers.Reshape(target_shape=[144, 1])(input_x)

    num_blocks = 2
    dilation_rates = (1, 2, 4, 8, 16, 32)

    for _ in range(num_blocks):
        for rate in dilation_rates:
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation=activation_function, dilation_rate=rate, padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)  

    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)

    # 输出层
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=[input_x], outputs=[x])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.0001))
    return model

import time
import pandas as pd
import tensorflow as tf

from tensorflow.keras import backend as K

# Define the shifted_relu activation function (if it's a custom function)
def shifted_relu(x, alpha=0.3):
    return K.maximum(alpha * x, x)

def create_model(activation_function):
    # This function should create your model using the activation function.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation_function, input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_record_loss_and_time(activation_function, num_trials=3):

    # Function to get a valid name for the activation function
    def get_activation_function_name(activation_function):
        if isinstance(activation_function, str):  # If it's a string like 'relu', 'elu', or 'shifted_relu'
            return activation_function
        elif isinstance(activation_function, tf.keras.layers.Layer):  # If it's an activation object like LeakyReLU
            class_name = activation_function.__class__.__name__  # Get class name (e.g., 'LeakyReLU')
            params = activation_function.get_config()  # Get any parameters (e.g., alpha for LeakyReLU)
            if 'alpha' in params:
                return f"{class_name}_alpha{params['alpha']}"  # For LeakyReLU, include alpha in the name
            return class_name
        elif callable(activation_function):  # If it's a custom function like shifted_relu
            return activation_function.__name__  # The name of the function
        else:
            raise ValueError("Unsupported activation function type")

    for trial in range(num_trials):
        print(f"\nStarting trial {trial+1} of {num_trials}...")

        # Create model with current activation function
        model = create_model(activation_function)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', restore_best_weights=True)
        
        # Measure training time
        start_time = time.time()
        history = model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=1000, batch_size=32, callbacks=[es], verbose=0)
        end_time = time.time()
        training_time = end_time - start_time

        # Get a valid name for the activation function
        activation_function_name = get_activation_function_name(activation_function)

        # Save the training history to a CSV file
        trial_history = history.history['loss']
        history_df = pd.DataFrame({'Epoch': range(1, len(trial_history) + 1), 'Loss': trial_history})
        history_df.to_csv(f"data/{activation_function_name}_trial_{trial+1}_loss_history.csv", index=False)

        # Save the training time to a TXT file
        with open(f"data/{activation_function_name}_trial_{trial+1}_training_time.txt", 'w') as f:
            f.write(f"{training_time}")
    
    return True

# Sample activations list with 'elu', 'relu', LeakyReLU, and custom shifted_relu
activation_functions = ['elu', 'relu', tf.keras.layers.LeakyReLU(alpha=0.3), shifted_relu]

# Call the function to train and record results for all activations
for activation in activation_functions:
    train_and_record_loss_and_time(activation, num_trials=3)


print("Training history and training times saved to CSV and TXT files.")
