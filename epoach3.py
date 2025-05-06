import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
file_path = 'data/selection.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Prepare features and labels
feature_columns = [col for col in data.columns if col != 'pathogen load']
X = data[feature_columns]
y = data['pathogen load']

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Define the 1D-CNN model
def create_model():
    input_x = tf.keras.layers.Input(shape=(X_train_scaled.shape[1],))
    x = tf.keras.layers.Reshape(target_shape=[X_train_scaled.shape[1], 1])(input_x)
    
    x = Conv1D(filters=7, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1)(x)
    
    model = Model(inputs=input_x, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to run training and save results
def run_training_and_save_results(run_number):
    # Initialize the model
    model = create_model()

    # List to store results for each epoch
    all_results = []

    # Initial training for the first set of epochs
    initial_epochs = 1
    history_1 = model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=initial_epochs, batch_size=32, verbose=0)

    # Evaluate after the first training
    y_pred_1 = model.predict(X_test_scaled)
    r2_1_test = r2_score(y_test, y_pred_1)
    y_pred_train_1 = model.predict(X_train_scaled)
    r2_1_train = r2_score(y_train, y_pred_train_1)

    # Save results after the first training
    train_loss_1, train_mae_1 = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss_1, test_mae_1 = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Run {run_number} - After {initial_epochs} Epochs:")
    print(f"Train Loss: {train_loss_1}, Train MAE: {train_mae_1}, Train R²: {r2_1_train}")
    print(f"Test Loss: {test_loss_1}, Test MAE: {test_mae_1}, Test R²: {r2_1_test}")

    # Save results for epoch 1
    epoch_1_results = {
        'epochs': initial_epochs,
        'train_loss': train_loss_1,
        'train_mae': train_mae_1,
        'train_r2': r2_1_train,
        'test_loss': test_loss_1,
        'test_mae': test_mae_1,
        'test_r2': r2_1_test
    }
    all_results.append(epoch_1_results)

    # Now continue training for the remaining epochs from 2 to 1000
    for epoch_count in range(2, 1001):
        # Continue training from the last saved epoch using the `initial_epoch` parameter
        history = model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=epoch_count, batch_size=32, verbose=0, initial_epoch=epoch_count - 1)

        # Evaluate after continuing training
        y_pred_test = model.predict(X_test_scaled)
        r2_test = r2_score(y_test, y_pred_test)
        
        y_pred_train = model.predict(X_train_scaled)
        r2_train = r2_score(y_train, y_pred_train)

        # Save results after each epoch
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
        
        print(f"Run {run_number} - After {epoch_count} Epochs:")
        print(f"Train Loss: {train_loss}, Train MAE: {train_mae}, Train R²: {r2_train}")
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test R²: {r2_test}")

        # Save the results for the current epoch
        epoch_results = {
            'epochs': epoch_count,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'train_r2': r2_train,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_r2': r2_test
        }
        all_results.append(epoch_results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save the results to a text file
    results_df.to_csv(f'data/model_results_run_{run_number}.txt', index=False)

# Run the training process 3 times and save results
for run_number in range(1, 4):
    run_training_and_save_results(run_number)
