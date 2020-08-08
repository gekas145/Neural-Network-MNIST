
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

def sigmoid(z):
    z = np.exp(-z)
    z += 1
    z = np.power(z, -1)
    return z


def sigmoid_prime(z):
    """  Derivative of sigmoid function """
    return np.power(sigmoid(z), 2) * np.exp(-z)


class Network:

    class Layer:
        """ Help class which represents layers of neural network """
        def __init__(self, n):
            self.bias = np.random.uniform(-5, 5, n)
            self.weights = np.array([0, 0, 0])
            self.activations = np.zeros(n)
            self.z = np.zeros(n)
            self.error = np.zeros(n)
            self.bias_der = np.zeros(n)
            self.weights_der = np.array([0, 0, 0])

        def change_bias(self, delta_bias):
            self.bias += delta_bias

        def change_weights(self, delta_weights):
            self.weights += delta_weights

        def set_bias(self, new_bias):
            self.bias = new_bias

        def set_weights(self, new_weights):
            self.weights = new_weights

        def set_activations(self, new_activations):
            self.activations = new_activations

        def set_z(self, new_z):
            self.z = new_z

        def set_error(self, new_error):
            self.error = new_error

        def set_bias_der(self, new_bias_der):
            self.bias_der = new_bias_der

        def set_weights_der(self, new_weights_der):
            self.weights_der = new_weights_der

        def __repr__(self):
            return str(self.bias) + "\n " + str(self.weights)





    def __init__(self, layers_sizes):
        self.imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
        self.lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
        self.layers = []
        for i in range(len(layers_sizes)):
            l = Network.Layer(layers_sizes[i])
            self.layers.append(l)
        for i in range(1, len(layers_sizes)):
            new_weights = np.zeros((layers_sizes[i], layers_sizes[i-1]))
            for j in range(len(new_weights)):
                new_weights[j] = np.random.uniform(-5, 5, len(new_weights[j]))
            self.layers[i].set_weights(new_weights)
            self.layers[i].set_weights_der(np.zeros((layers_sizes[i], layers_sizes[i - 1])))

    def Print(self):
        for i in range(len(self.layers)):
            print(f"Layer {i+1}")
            print(self.layers[i])



    def feedforward(self, input):
        """  Analyses input and returns so called decision """
        self.layers[0].set_activations(input)
        for i in range(1, len(self.layers)):
            z = np.dot(self.layers[i].weights, self.layers[i-1].activations) + self.layers[i].bias
            self.layers[i].set_z(z)
            z = sigmoid(z)
            self.layers[i].set_activations(z)

        return self.layers[len(self.layers) - 1].activations

    def backprop(self, input, output, eta = 0.1):
        pass
     #   res = self.feedforward(input)
     #   self.layers[len(self.layers) - 1].set_error((res - output) * sigmoid_prime(self.layers[len(self.layers) - 1].z))







net = Network([3, 5, 10])
print(net.feedforward(np.array([0.1, 0.3, 0.56])))





imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
imagearray = imagearray / 255
for i in range(5):
    print(lablesarray[i])

for i in imagearray[0]:
    print(i)

plt.imshow(imagearray[0], cmap=plt.cm.binary)
plt.show()

