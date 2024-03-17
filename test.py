import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scripts.data_prep_GMSC import load_data
from sklearn.model_selection import train_test_split

# Assuming you have your dataset loaded into X, y
path = ".\GiveMeSomeCredit\cs-training.csv"

X,Y, data  = load_data(path)

# X = X[:5]
# Y = Y[:5]

n = X.shape[0]
d = X.shape[1] - 1

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the architecture of the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(11,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

pred = model.predict(X_test)

## check if acc is calculated correct 

print( np.mean((pred > 0.5) == y_test))  
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)