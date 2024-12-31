import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Data Loading (Replace 'data.csv' with the actual dataset path)
data = pd.read_csv('data.csv')  # Assume columns: ["temperature", "humidity", "pressure", ..., "rainfall"]

# 2. Data Preprocessing
X = data.drop(columns=['rainfall'])  # Features
y = data['rainfall']  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. ANN Model Definition
def build_ann():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 4. DNN Model Definition
def build_dnn():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 5. Train and Evaluate ANN
ann_model = build_ann()
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# ANN Evaluation
y_pred_ann = ann_model.predict(X_test)
ann_mse = mean_squared_error(y_test, y_pred_ann)
ann_r2 = r2_score(y_test, y_pred_ann)
print(f"ANN - MSE: {ann_mse}, R2: {ann_r2}")

# 6. Train and Evaluate DNN
dnn_model = build_dnn()
dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# DNN Evaluation
y_pred_dnn = dnn_model.predict(X_test)
dnn_mse = mean_squared_error(y_test, y_pred_dnn)
dnn_r2 = r2_score(y_test, y_pred_dnn)
print(f"DNN - MSE: {dnn_mse}, R2: {dnn_r2}")

# 7. Compare Results
print("\nModel Comparison:")
print(f"ANN - MSE: {ann_mse}, R2: {ann_r2}")
print(f"DNN - MSE: {dnn_mse}, R2: {dnn_r2}")
