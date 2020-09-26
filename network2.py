import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import json




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

    def __init__(self, layers_sizes):
        self.imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
        self.lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
        self.testimage = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/t10k-images.idx3-ubyte')
        self.testlable = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/t10k-labels.idx1-ubyte')
        self.imagearray = np.true_divide(self.imagearray, 255)
        self.testimage = np.true_divide(self.testimage, 255)
        self.layers = []
        self.learn_progress = np.array([0, 0, 0])  # for learn progress plotting
        for i in range(len(layers_sizes)):
            l = Network.Layer(layers_sizes[i])
            self.layers.append(l)
            if i != 0:
                self.layers[i].weights = np.random.randn(layers_sizes[i], layers_sizes[i-1])
                self.layers[i].weights_der = np.zeros((layers_sizes[i], layers_sizes[i-1]))

    @staticmethod
    def sigmoid(z):
        """ The sigmoid function """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """  Derivative of the sigmoid function """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    @staticmethod
    def load_data(x):
        """ Reshapes matrix which represents an image into single 1-D array """
        return x.reshape(784)


    def feedforward(self, input):
        """  Analyses input and returns so called decision """
        self.layers[0].activations = input
        for i in range(1, len(self.layers)):
            z = np.dot(self.layers[i].weights, self.layers[i - 1].activations) + self.layers[i].bias
            self.layers[i].z = z
            self.layers[i].activations = Network.sigmoid(z)

        return self.layers[-1].activations


    def update_mini_batch(self, n, m):
        return self.imagearray[n: n + m], self.lablesarray[n: n + m]



    def backprop(self, input, output):
        res = self.feedforward(input)
        self.layers[-1].error = res - output
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].bias_der += self.layers[i].error
            for j in range(len(self.layers[i].error)):
                self.layers[i].weights_der[j] += self.layers[i-1].activations * self.layers[i].error[j]
            if i == 1:
                break
            self.layers[i-1].error = np.dot(np.transpose(self.layers[i].weights), self.layers[i].error) * Network.sigmoid_prime(self.layers[i-1].z)





    def gradient_descent(self, eta = 0.5, m = 10, epochs = 30, lmbda = 5, regular = True, large_weights = False):
        if not large_weights:
            self.weights_init()
        n = 0  # number of current mini-batch
        # might 50 000 instead of len(self.imagearray)
        k = int(len(self.imagearray)/(m * epochs))  # quantity of mini-batches in one epoch
        self.learn_progress = np.zeros(epochs)
        for i in range(epochs):

            rate = 0  # right answers count

            for j in range(len(self.testimage)):
                ans = self.feedforward(Network.load_data(self.testimage[j]))
                if np.amax(ans) == ans[self.testlable[j]]:
                    rate += 1
            self.learn_progress[i] = rate/100

            for t in range(k):
                tmp = self.update_mini_batch(n, m)
                n += m
                images = tmp[0]
                lables = tmp[1]

                for image, label in zip(images, lables):
                    y = np.zeros(10)
                    y[label] = 1
                    self.backprop(Network.load_data(image), y)

                for j in range(1, len(self.layers)):
                    self.layers[j].bias_der = np.true_divide(self.layers[j].bias_der, m)
                    self.layers[j].weights_der = np.true_divide(self.layers[j].weights_der, m)
                    self.layers[j].bias -= self.layers[j].bias_der * eta
                    if regular:
                        self.layers[j].weights -= self.layers[j].weights_der * eta + np.true_divide(self.layers[j].weights * eta * lmbda, len(self.imagearray))
                    else:
                        self.layers[j].weights -= self.layers[j].weights_der * eta
                    self.layers[j].bias_der *= 0
                    self.layers[j].weights_der *= 0

        self.test()


    def weights_init(self):
        """ Smart weights initializing with mean 0 and std deviation 1/sqrt(num_of_inputs) """
        for i in range(1, len(self.layers)):
            self.layers[i].weights = np.random.normal(0, 1/np.sqrt(len(self.layers[i].weights[0])), [len(self.layers[i].weights), len(self.layers[i].weights[0])])




    def test(self):
        rate = 0
        for image, label in zip(self.testimage, self.testlable):
            # y = np.zeros(10)
            # y[label] = 1
            # rate += np.sum(np.power(self.feedforward(load_data(image)) - y, 2))
            # if np.sum(np.power(self.feedforward(load_data(image)) - y, 2)) < 0.2:
            #     rate += 1
            ans = self.feedforward(Network.load_data(image))
            if np.amax(ans) == ans[label]:
                rate += 1
        print(f"rate is {rate}/{len(self.testlable)}")
        # print(f"Mean square error: {rate/len(self.testlable)}")

    def save(self, filename='C:/Users/yevhe/Downloads/samples/network.json'):
        """ Saves trained network to file with path `filename` """
        weights_list = [[0] for i in range(len(self.layers) - 1)]
        bias_list = [[0] for i in range(len(self.layers) - 1)]

        for i in range(1, len(self.layers)):
            weights_list[i-1] = self.layers[i].weights.tolist()
            bias_list[i-1] = self.layers[i].bias.tolist()

        data = {"weights": weights_list,
                "bias": bias_list}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    @staticmethod
    def load(filename='C:/Users/yevhe/Downloads/samples/network.json'):
        """ Loads trained network from file with path `filename` """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        net = Network([784, 100, 10])

        for i in range(1, len(net.layers)):
            net.layers[i].weights = np.array(data["weights"][i-1])
            net.layers[i].bias = np.array(data["bias"][i-1])

        return net





def main():
    # net = Network([784, 100, 10])
    # net.gradient_descent()
    # net.save()
    net2 = Network.load()
    for i in range(10):
        print(net2.feedforward(Network.load_data(net2.testimage[i])))
        print(net2.testlable[i])
        print("------------------------")





if __name__ == '__main__':
    main()