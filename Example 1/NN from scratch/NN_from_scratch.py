import numpy as np

class SimpleNN:
    def __init__(self):
        self.W1 = np.array([[1, 1], [1, -1], [-1, 1]], dtype=np.float64)
        self.b1 = np.array([0, 1, -1], dtype=np.float64)

        self.W2 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, 1]], dtype=np.float64)
        self.b2 = np.array([0, -1, 1], dtype=np.float64)

        self.W3 = np.array([[1, 1, 1], [1, -1, -1]], dtype=np.float64)
        self.b3 = np.array([0, 1], dtype=np.float64)

    def forward(self, x):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = np.maximum(0, z1)

        z2 = np.dot(self.W2, a1) + self.b2
        a2 = np.maximum(0, z2)

        z3 = np.dot(self.W3, a2) + self.b3
        a3 = np.maximum(0, z3)

        return a3

    def backpropagate(self, x, y, learning_rate=0.01):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = np.maximum(0, z1)

        z2 = np.dot(self.W2, a1) + self.b2
        a2 = np.maximum(0, z2)

        z3 = np.dot(self.W3, a2) + self.b3
        a3 = np.maximum(0, z3)

        # loss (mean squared error)
        loss = np.mean((a3 - y) ** 2)

        # Backpropagation to calculate gradients
        delta3 = a3 - y  # output layer error
        delta2 = np.dot(self.W3.T, delta3) * (z2 > 0)  # error at second hidden layer
        delta1 = np.dot(self.W2.T, delta2) * (z1 > 0)  # error at first hidden layer

        # Update weights and biases
        self.W3 -= learning_rate * np.outer(delta3, a2)
        self.b3 -= learning_rate * delta3

        self.W2 -= learning_rate * np.outer(delta2, a1)
        self.b2 -= learning_rate * delta2

        self.W1 -= learning_rate * np.outer(delta1, x)
        self.b1 -= learning_rate * delta1

        # Print updated weights and biases
        print("\nUpdated parameters after backpropagation:")
        print("W1:")
        print(self.W1)
        print("b1:")
        print(self.b1)
        print("\nW2:")
        print(self.W2)
        print("b2:")
        print(self.b2)
        print("\nW3:")
        print(self.W3)
        print("b3:")
        print(self.b3)

        return loss


# Execution
network = SimpleNN()
input = np.array([0.5, 0.5])
y = np.array([2, 1]) # target

output = network.forward(input)
print(f"Output: {output}\n")

loss = network.backpropagate(input, y)
