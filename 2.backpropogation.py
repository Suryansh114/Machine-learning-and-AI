#Mathematically solving a back propogation question with example try running it for better understanding

import numpy as np

# Activation function and its derivative
def sigmoid(n):
    return 1 / (1 + np.exp(-n))

def sigmoid_der(n):
    return n * (1 - n)  # correct derivative of sigmoid output

# Loss and its derivative
def loss(pred, actual):
    return np.mean(np.square(pred - actual))

def loss_der(pre, actual):
    return 2 * (pre - actual)

#We have taken 4 input examples and we can solve them altogether using np library
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])
#This is the calculated output which is given to us in the NN itself
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

#These are the weights of the arrow from input layer to hidden layer
W1 = np.array([
    [1, 2],
    [3, 4]
])  

#These are the bias in the hidden layer
b1 = np.array([[5, 6]])  

#These are the weights from hidden layer to output layer
W2 = np.array([
    [7],
    [8]
])  

#This is the output of the bias layer
b2 = np.array([[9]])
np.random.seed(0)

input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# Training
epochs = 10000 #epoch are the number of iterations that are required to get precise results
lr = 0.1 # lr = learning rate is how much the results or weights are adjusted in every iteration

for epoch in range(epochs):
    Z1 = np.dot(X, W1) + b1 # here using np we multiply the X matrix (input matrix) with weight and add constant weights(called biases)
    A1 = sigmoid(Z1) #Then on the output we form the result as the sigmoid of Z1

    Z2 = np.dot(A1, W2) + b2 #Similar to Z1
    A2 = sigmoid(Z2) #similar to A1

    loss = np.mean((y - A2)**2) #This is the method to calculate loss


#Now the maths behind back propogation will be mentioned in the view file
    dA2 = A2 - y        
    dZ2 = dA2 * sigmoid_der(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_der(A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
#This is subtracting the error from the initial weights and using the learning rate to alter the affect of it
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# Final predictions
print("\nPredictions after training:")
print(np.round(A2, 2))
