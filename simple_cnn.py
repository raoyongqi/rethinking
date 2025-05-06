import tensorflow as tf

import pandas as pd


import numpy as np


from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape
file_path = 'data/merged_all.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)


feature_columns = [col for col in data.columns if col != 'Pathogen Load']

from sklearn.model_selection import train_test_split

X = data[feature_columns]


y = data['Pathogen Load']

from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
input_x = tf.keras.layers.Input(shape=(144,))
x = tf.keras.layers.Reshape(target_shape=[144, 1])(input_x)

# Add a Convolutional Layer
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)

# Add a Max Pooling Layer
x = MaxPooling1D(pool_size=2)(x)

# Flatten the data for the fully connected layer
x = Flatten()(x)

# Add Dense Layer
x = Dense(128, activation='relu')(x)

# Output Layer
output = Dense(1)(x)  # No activation function here because it's regression

# Create the model
model = Model(inputs=input_x, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary of the model
model.summary()

history = model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=1000, batch_size=32)

# # Predict on the test set
# y_pred = model.predict(X_test_scaled)

# # Calculate R-squared (R²) score
# r2 = r2_score(y_test, y_pred)

# print(f"R² Score on Test Set: {r2}")


import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import plot_model
# Plot and save the model architecture
plot_model(model, to_file='data/a_model_architecture.png', show_shapes=True, show_layer_names=True)

# # Display the image using matplotlib
# img = mpimg.imread('model_architecture.png')
# imgplot = plt.imshow(img)
# plt.axis('off')  # Hide axes
# plt.show()