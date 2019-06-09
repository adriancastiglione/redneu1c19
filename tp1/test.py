from multilayerperceptron import MultilayerPerceptron
from perceptron import Perceptron
import numpy as np


np.random.seed(0)

xor_data = [[1,1], [1,0], [0,1], [0,0]]
xor_answ = [0, 1, 1, 0]

mlp = MultilayerPerceptron("sigmoid", [(2, "sigmoid")])
mlp.fit(xor_data, xor_answ, learning_rate=1.1, epochs = 5000)

print("(0,0) -> {}".format(mlp.predict([[0,0]])))
print("(1,0) -> {}".format(mlp.predict([[1,0]])))
print("(0,1) -> {}".format(mlp.predict([[0,1]])))
print("(1,1) -> {}".format(mlp.predict([[1,1]])))

print(mlp.predict(xor_data))


or_data = [[1,1], [1,0], [0,1], [0,0]]
or_answ = [1, 1, 1, 0]

p = Perceptron(2)
p.fit(or_data, or_answ)

print("")
print("")

print("(0,0) -> {}".format(p.predict([[0,0]])))
print("(1,0) -> {}".format(p.predict([[1,0]])))
print("(0,1) -> {}".format(p.predict([[0,1]])))
print("(1,1) -> {}".format(p.predict([[1,1]])))

print(p.predict(or_data))
