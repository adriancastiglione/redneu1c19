
import numpy as np
from math import exp
import copy
from collections import namedtuple

layer_data = namedtuple("LayerData", "neurons_number activation_function")

class MultilayerPerceptron:


	activation_functions = {
		"sigmoid" : lambda x: 1 / (1 + np.exp(-x)),
		"tanh": lambda x: np.tanh(x), #(np.exp(x) - (np.exp(-x)))/(np.exp(x) + (np.exp(-x))),
		"identity": lambda x: x
	}


	activation_functions_derivative = {
		"sigmoid" : lambda h, v: v*(1-v), #exp((-1)*x) / (1 + exp(-1*x))**2,
		"tanh": lambda h, v: 1 - v**2,#(exp(x) - (exp(-1*x)))/(exp(x) + (exp(-1*x))),
		"identity": lambda h, v: 1
	}


	#hidden_layers is an iterable with the sizes of the hidden layers, where the i-th element is the size of
	#the i-th hidden layer. The MLP will add input and output layers according to training data shape. 
	def __init__(self, activation, hidden_layers = []):

		self.needs_training = True
		self.hidden_layers = []
		self.output_layer_activation = activation
		self._weights_not_initialized = True

		for x in hidden_layers:
			layer = self._build_layer(x)
			self.hidden_layers.append(layer)


	def _build_layer(self, layer):
		NEURONS_NUMBER = 0
		ACTIVATION = 1

		if layer[NEURONS_NUMBER] <= 0:
			raise ValueError("A layer must have at least one neuron")

		return layer_data(layer[NEURONS_NUMBER], layer[ACTIVATION])


	def add_activation_function(self, name, function, derivative):
		if name in self.activation_functions:
			raise ValueError("activation function already exists")

		self.activation_functions[name] = function
		self.activation_functions_derivative[name] = derivative


	def remove_activation_function(self, name):
		del self.activation_functions[name]
		del self.activation_functions_derivative[name]


	def predict(self, patterns):
		if self.needs_training:
			raise RuntimeError("You need to train the network before infering values")

		output = []
		
		for u in patterns:
			current_prediction = self._predict(u)
			y =  current_prediction if current_prediction.size > 1 else current_prediction[0]
			output.append(y)

		return np.array(output)


	#pre: weights are already initialized
	def _predict(self, pattern):

		p = np.atleast_1d(pattern)
		x = np.insert(p, 0, 1.)
		self.net_inputs_cache = [p]
		self.outputs_cache = [x]

		#for k in range(1, len(self.layers) + 1 + 1):
		all_layers = self.hidden_layers + [self._build_layer((1, self.output_layer_activation))] #output layer dimention not important non disponible in current scope
		for k in range(1, len(all_layers) + 1):
			layer = all_layers[k-1]
			g = np.vectorize(self.activation_functions[layer.activation_function])
			h = np.matmul(self.W[k], x)
			v = g(h)
			x = np.insert(v, 0, 1.)
			self.net_inputs_cache.append(h)
			self.outputs_cache.append(x)

		return v
		

	def fit(self, patterns, answers, learning_rate = 0.2, epochs = 1000, momentum=0, refitting = False, adaptative_learning_rate=None):
	

		a = 0
		b = 0
		dE = 0
		previousE = 0

		if adaptative_learning_rate is not None:
			a = adaptative_learning_rate[0]
			b = adaptative_learning_rate[1]

		if len(patterns) == 0 or len(answers) == 0:
			raise ValueError("Patterns nor answers can be zero")

		if len(patterns) != len(answers):
			raise ValueError("Not matching patterns and output lengths")

		input_dimention = len(np.atleast_1d(patterns[0]))
		output_dimention = len(np.atleast_1d(answers[0]))

		if not refitting or self._weights_not_initialized:
			self._initialize_weights(input_dimention, output_dimention)
		
		layers = self.hidden_layers + [self._build_layer((output_dimention, self.output_layer_activation))]
		M = len(layers)

		for epoch in range(0, epochs):

			for u in range(0, len(patterns)):

				current_pattern = patterns[u]
				expected_output = answers[u]
				self.delta = [0 for x in range(0, M + 1)]
				delta = self.delta 
				

				dg = np.vectorize(self.activation_functions_derivative[self.output_layer_activation])

				current_model_output = self._predict(current_pattern)
				output_difference = expected_output - current_model_output
				delta[M] = dg(self.net_inputs_cache[M], self.outputs_cache[M][1:])*output_difference
				
				E = np.dot(output_difference, output_difference) / 2
				dE = E - previousE 
				if dE < 0:
					learning_rate += a 
				elif dE > 0:
					learning_rate -= b*learning_rate

				previousE = E

				for m in range(M, 1, -1):
					dg = np.vectorize(self.activation_functions_derivative[layers[m-1].activation_function])
					delta[m-1] = dg(self.net_inputs_cache[m-1], self.outputs_cache[m-1][1:])*np.matmul(np.transpose(self.W[m][:, 1:]), delta[m])

				for m in range(1, M + 1):
					v_m_1 = self.outputs_cache[m-1]
					d = delta[m]
					dW_current = learning_rate * np.matmul(np.reshape(d, (d.size, 1)), np.reshape(v_m_1, (1, v_m_1.size)))
					self.W[m] += dW_current + momentum * self.dW[m]
					self.dW[m] = dW_current

		self.needs_training = False


	def _initialize_weights(self, input_dimention, output_dimention):

		self.W = [[]]
		self.dW = [[]]

		layers = [self._build_layer((input_dimention, "identity"))] + self.hidden_layers + [self._build_layer((output_dimention, self.output_layer_activation))]
		
		for k in range(1, len(layers)):
			previous_layer_dimention = layers[k-1].neurons_number
			current_layer_dimention = layers[k].neurons_number
			w_k = np.random.normal(0, 0.1, size = (current_layer_dimention, previous_layer_dimention + 1) )
			dw_k = np.zeros((current_layer_dimention, previous_layer_dimention + 1))
			self.W.append(w_k)
			self.dW.append(dw_k)

		self._weights_not_initialized = False








