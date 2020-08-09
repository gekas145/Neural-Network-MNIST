
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

def sigmoid(z):
    """ The sigmoid function """
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    """  Derivative of the sigmoid function """
    return sigmoid(z)*(1-sigmoid(z))

def load_data(x):
    """ Reshapes matrix which represents an image into single 1-D array """
    return x.reshape(784)


class Network:

    class Layer:
        """ Help class which represents layers of neural network """
        def __init__(self, n):
            self.bias = np.random.randn(n)
            self.weights = np.array([0, 0, 0])
            self.activations = np.zeros(n)
            self.z = np.zeros(n)
            self.error = np.zeros(n)
            self.bias_der = np.zeros(n)
            self.weights_der = np.array([0, 0, 0])


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

        def change_bias_der(self, delta_bias_der):
            self.bias_der += delta_bias_der

        def change_weights_der(self, delta_weights_der, n):
            self.weights_der[n] += delta_weights_der

        def __repr__(self):
            return "Bias \n" + str(self.bias) + "\n " + "\n" + "Weights \n" + str(self.weights)





    def __init__(self, layers_sizes):
        self.imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
        self.lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
        self.testimage = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/t10k-images.idx3-ubyte')
        self.testlable = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/t10k-labels.idx1-ubyte')
        self.imagearray = np.true_divide(self.imagearray, 255)
        self.testimage = np.true_divide(self.testimage, 255)
        self.layers = []
        for i in range(len(layers_sizes)):
            l = Network.Layer(layers_sizes[i])
            self.layers.append(l)
            if i != 0:
                self.layers[i].set_weights(np.random.randn(layers_sizes[i], layers_sizes[i-1]))
                self.layers[i].set_weights_der(np.zeros((layers_sizes[i], layers_sizes[i-1])))


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

        return self.layers[-1].activations



    def update_mini_batch(self, n, m):
        return self.imagearray[n: n + m], self.lablesarray[n: n + m]



    def backprop(self, input, output):
        res = self.feedforward(input)
        # print(self.layers[-1].error)
        # print("\n \n \n")
        self.layers[-1].set_error((res - output) * sigmoid_prime(self.layers[-1].z))
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].change_bias_der(self.layers[i].error)
            for j in range(len(self.layers[i].error)):
                self.layers[i].change_weights_der(self.layers[i-1].activations * self.layers[i].error[j], j)
            if i == 1:
                break
            self.layers[i-1].set_error(np.dot(np.transpose(self.layers[i].weights), self.layers[i].error) * sigmoid_prime(self.layers[i-1].z))



    def gradient_descent(self, eta = 3, m = 100, epochs = 600):
        n = 0  # number of current mini-batch
        for i in range(epochs):
            tmp = self.update_mini_batch(n, m)
            n += m
            images = tmp[0]
            lables = tmp[1]
            for image, label in zip(images, lables):
                y = np.zeros(10)
                y[label] = 1
                self.backprop(load_data(image), y)
            for j in range(1, len(self.layers)):
                self.layers[j].bias_der = np.true_divide(self.layers[j].bias_der, m)
                self.layers[j].weights_der = np.true_divide(self.layers[j].weights_der, m)
                self.layers[j].bias -= self.layers[j].bias_der * eta
                self.layers[j].weights -= self.layers[j].weights_der * eta
                self.layers[j].bias_der *= 0
                self.layers[j].weights_der *= 0
        self.test()



    def test(self):
        rate = 0
        for image, label in zip(self.testimage, self.testlable):
            y = np.zeros(10)
            y[label] = 1
            if np.sum(np.power(self.feedforward(load_data(image)) - y, 2)) < 0.5:
                rate += 1
        print(f"rate is {rate}/10000")














net = Network([784, 15, 10])
net.gradient_descent()



# imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
# lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
# imagearray = imagearray / 255
# for i in range(5):
#     print(lablesarray[i])
#
# for i in imagearray[0]:
#     print(i)
#
# plt.imshow(imagearray[0], cmap=plt.cm.binary)
# plt.show()

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# c = a
# b = np.array([1, 2, 3])
# print(np.dot(a, b))
# print(sigmoid(np.array([1, 2, 3])))
# print(np.random.rand(3, 5))
# print(a[0] + b)
# a *= 0
# print(a)

