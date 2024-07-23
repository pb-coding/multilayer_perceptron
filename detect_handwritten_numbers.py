import tensorflow as tf
import numpy as np

# Define the model: Multi-layer Perceptron (MLP) with 2 hidden layers
# It has 13,002 parameters (weights & biases) to learn
model = tf.keras.models.Sequential([
    # purpose: input layer - converts 2D array to 1D array - 28x28 = 784 - no parameters to learn
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # purpose: hidden layer with 16 nodes - 784 * 16 + 16 = 12560 parameters to learn
    tf.keras.layers.Dense(16, activation='relu'),
    
    # purpose: Dropout layer: Randomly sets input units to 0 (set to 20% Dropout) - helps prevent overfitting.
    tf.keras.layers.Dropout(0.2),
    
    # purpose: hidden layer with 16 nodes - 16 * 16 + 16 = 272 parameters to learn
    tf.keras.layers.Dense(16, activation='relu'),
    
    # purpose: output layer with 10 nodes - 16 * 10 + 10 = 170 parameters to learn - 
    # we use softmax activation function to convert the raw logit output to probabilities
    tf.keras.layers.Dense(10, activation='softmax')
])


# load mnist dataset
mnist = tf.keras.datasets.mnist

# split dataset into training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
# we divide by 255 to scale the pixel values to a range of 0 to 1
# this improves the performance of the model
x_train, x_test = x_train / 255.0, x_test / 255.0

# define loss function
# from_logits=False because we have used softmax activation function in the output layer	
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# train the model on the training data
model.fit(x_train, y_train, epochs=5)

# evaluate the model on the test data
model.evaluate(x_test,  y_test, verbose=2)

print("Labels of first 5 images from test data: " + str(y_test[:5]))
predictions = model(x_test[:5])

# round the predictions to 4 decimal places to make it easier to read
np.set_printoptions(precision=4, suppress=True)
print("Predictions of first 5 images from test data: " + str(predictions))
