import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.learning_rate=learning_rate
        self.weights=np.zeros(input_size)
        self.bias=0.0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_probability(self, X):
        X=np.array(X)
        z=np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        probability=self.predict_probability(X)

        return [1 if p >= 0.5 else 0 for p in probability]