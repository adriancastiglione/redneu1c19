
import numpy as np
import math

class Perceptron:

	def __init__(self, entradas, salidas, activacion):

		self._entradas = entradas
		self._salidas = salidas
		self._activacion = activacion
		self._pesos = np.random.rand(entradas + 1, salidas)

	def _g(self, x):

		if self._activacion == "sigmoid":
			return 1/(1 + math.exp((-1)*x))

		if self._activacion == "relu":
			return max(0, x)

		if self._activacion == "step":
			return 1 if x > 0.5 else 0


	def _calcularSalidaPara(self, X):

		outputs = np.zeros(self._salidas)

		for Oi in range(0, self._salidas):
			sum = 0
			for Ei in range(1, self._entradas + 1):
				sum +=  self._pesos[Ei][Oi] * X[Ei - 1]

			outputs[Oi] = self._g(sum - self._pesos[0][Oi]) 

		return outputs


	def fit(self, X, Y, epocas = 1000, eta = 0.1):

		X = np.reshape(X, (len(X), self._entradas))

		for e in range(0, epocas):
			for i in range(0, len(X)):
				for j in range(0, self._salidas):
					Oj = self._calcularSalidaPara(X[i])
					update = eta * (Y[i] - Oj)
					self._pesos[1:, j] += update * X[i]
					self._pesos[0, j] += update * -1


	def predict(self, x):
		return self._calcularSalidaPara(x)


if __name__ == "__main__":

	#funcion xor
	X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
	Y = np.array([0, 1, 1, 1])

	p = PerceptronSimple(entradas = 2, salidas = 1, activacion = "sigmoid")
	p.fit(X, Y, epocas = 1000, eta = 0.15)

