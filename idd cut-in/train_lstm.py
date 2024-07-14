import numpy as np
import pandas as pd
import tensorflow as tf
from data_preparation import prepare_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Prepare data
X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_data()

# Define model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh', return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, verbose=1)

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,  # Adjust the number of epochs based on your needs
                    batch_size=32,  # Adjust the batch size based on your needs
                    callbacks=[reduce_lr])

# Save model
model.save('lstm_model_idd.h5')
