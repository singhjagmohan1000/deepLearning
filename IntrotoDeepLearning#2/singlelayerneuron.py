from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		random.seed(1)
		self.synaptic_weights = 2 * random.random((3,1))-1

	#sigmoid function
	def __sigmoid(self, x):
		return 1 / (1+exp(-x))

	#derivative of sigmoid function
	def __sigmoid_derivative(self,x):
		return x * (1-x)

	#training
	def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
	    for i in range(number_of_iterations):
	        output = self.think(training_set_inputs)

	        #calculate error
	        error = training_set_outputs - output

	        adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

	        #Adjust Weight
	        self.synaptic_weights += adjustment

    #neural network thinks
	def think(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))
if __name__ == '__main__':

	#initialise a single neuron neural network
	neural_network = NeuralNetwork()

	print("Random starting synaptic weights:")
	print(neural_network.synaptic_weights)

	# The training set
	training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print("New synaptic weights after training:")
	print(neural_network.synaptic_weights)

	#Test the neural Network
	print("Considering new Situation [1,0,0]->: ")
	print(neural_network.think(array([1,0,0])))