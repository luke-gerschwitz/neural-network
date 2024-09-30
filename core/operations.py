from constants import Mathematics

def sigmoid(x):
    return 1 / (1 + pow(Mathematics.EULER, -x))

def tanh(x):
    euler = Mathematics.EULER
    return (pow(euler, x) - pow(euler, -x)) / (pow(euler, x) + pow(euler, -x))