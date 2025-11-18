import numpy as np
from perceptron import Perceptron
from sklearn.metrics import accuracy_score
from data import get_training_data

def main(X, y):
    preds = np.array(model.predict(X))
    print("Accuracy:", accuracy_score(y, preds))