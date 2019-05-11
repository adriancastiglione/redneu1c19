
import numpy as np
from math import exp


NUM_NEURONS = 0
ACTIVATION = 1

class MultilayerPerceptron:


	activationFunction = {
		"sigmoid" : lambda x: 1 / (1 + exp(-1*x)),
		"relu" : lambda x: max(0, x),
		"tanh": lambda x: (exp(x) - (exp(-1*x)))/(exp(x) + (exp(-1*x))),
		"slope": lambda x: 1 if x > 0 else 0
	}


	#layers is a list of tuples (a,b) such that len(layers) is total amount of layers in
	#the perceptron, where a is the number of neurons and b the activation function.
	#pre: len(layers) > 1
	def __init__(self, layers):

		if not len(layers) > 1:
			raise ValueError

		
		
		self.layers = layers
		self.W = []
		self.activation =[]

		for x in range(0, len(layers) - 1):
			firstlayer = layers[x]
			secondlayer = layers[x+1]

			self.W.append(np.random.rand(firstlayer[NUM_NEURONS] + 1, secondlayer[NUM_NEURONS]))

		for x in range(0, len(layers)):
			self.activation.append(self.activationFunction[layers[x][ACTIVATION]])

		
	#pre: shape(x) == layers[0][NUM_NEURONS]
	def calcular(self, x):

		inputPattern = x
		outputPattern = []

		for currentLayer in range(0, len(self.layers) - 1):
			nextLayer = currentLayer + 1
			for x in range(0, self.layers[nextLayer][NUM_NEURONS]):
				pattern = [1] + inputPattern
				netInput = np.dot(pattern, self.W[currentLayer][:, x])
				outputPattern.append(self.activation[nextLayer](netInput))
				inputPattern = outputPattern

		return outputPattern if len(outputPattern) > 1 else outputPattern[0]


