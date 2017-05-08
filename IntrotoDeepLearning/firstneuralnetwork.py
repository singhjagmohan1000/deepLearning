import  numpy as np


class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3,1)) -1

    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def __sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights +=adjustment

    def predict(self,inputs):
        return self.__sigmoid(np.dot(inputs,self.synaptic_weights))




if __name__ == "__main__":

    #Initialize a single neuron network
    neural_network = NeuralNetwork()
    print('Random weights:')
    print(neural_network.synaptic_weights)

    #Training set
    training_set_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = np.array([[0, 1,  1, 0]]).T

    #Trains the training set

    neural_network.train(training_set_inputs,training_set_outputs,10000)

    print("New Weights")
    print(neural_network.synaptic_weights)


    #Prediction
    print("Output")
    print(neural_network.predict(np.array([1,0,0])))







