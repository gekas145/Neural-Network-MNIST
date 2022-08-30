import numpy as np
from load_mnist import load_mnist
import matplotlib.pyplot as plt
import json


class Network:
    class Layer:
        """ Help class which represents layers of neural network """

        def __init__(self, n, input_layer=False):
            if input_layer:
                k = 0  # input layer has no weights nor biases
            else:
                k = n
            self.bias = np.random.randn(k)
            self.weights = None  # will be initialized at network creation
            self.activations = np.zeros(n)
            self.raw = np.zeros(k)
            self.error = np.zeros(k)
            self.bias_der = np.zeros(k)  # bias derivative
            self.weights_der = None  # weights derivative, will be initialized at network creation

    def __init__(self, layers_sizes, smart_weights=True):
        self.layers = [Network.Layer(layers_sizes[0], True)]
        self.learn_progress = None
        for i in range(1, len(layers_sizes)):
            self.layers.append(Network.Layer(layers_sizes[i]))
            self.layers[i].weights_der = np.zeros((layers_sizes[i], layers_sizes[i - 1]))
            self.layers[i].weights = np.random.randn(layers_sizes[i], layers_sizes[i - 1])
            if smart_weights:
                self.layers[i].weights *= 1 / np.sqrt(self.layers[i].weights.shape[1])

    @staticmethod
    def sigmoid(raw):
        """ The sigmoid function """
        return 1 / (1 + np.exp(-raw))

    @staticmethod
    def sigmoid_prime(raw):
        """  Derivative of the sigmoid function """
        return Network.sigmoid(raw) * (1 - Network.sigmoid(raw))

    @staticmethod
    def get_mini_batch(X, y, n, m):
        return X[n: min(n + m, len(X))], y[n: min(n + m, len(y))]

    def feedforward(self, input):
        """  Analyses input and returns decision """
        self.layers[0].activations = input
        for i in range(1, len(self.layers)):
            raw = np.dot(self.layers[i].weights, self.layers[i - 1].activations) + self.layers[i].bias
            self.layers[i].raw = raw
            self.layers[i].activations = Network.sigmoid(raw)

        return self.layers[-1].activations

    def predict(self, input):
        ans = self.feedforward(input)
        return np.argmax(ans)

    def backprop(self, input, output):
        res = self.feedforward(input)
        self.layers[-1].error = res - output
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].bias_der += self.layers[i].error
            for j in range(len(self.layers[i].error)):
                self.layers[i].weights_der[j] += self.layers[i - 1].activations * self.layers[i].error[j]
            if i == 1:
                break
            self.layers[i - 1].error = np.dot(np.transpose(self.layers[i].weights),
                                              self.layers[i].error) * Network.sigmoid_prime(self.layers[i - 1].raw)

    def gradient_descent(self, X, y,
                         X_test=None, y_test=None,
                         eta=0.5, mini_batch_size=10, epochs=30, lmbda=5):
        self.learn_progress = np.zeros(epochs + 1)
        mini_batch_num = len(X) // mini_batch_size + 1
        for i in range(epochs):
            accuracy_str = ''
            if X_test is not None and y_test is not None:
                accuracy = self.test(X_test, y_test)
                accuracy_str += f', accuracy: {accuracy}'
                self.learn_progress[i] = accuracy
            print(f'[EPOCH {i + 1}/{epochs}]' + accuracy_str)
            current_mini_batch_start = 0  # start index of current mini-batch
            for t in range(mini_batch_num):
                mini_batch = Network.get_mini_batch(X, y, current_mini_batch_start, mini_batch_size)
                current_mini_batch_start += mini_batch_size
                images = mini_batch[0]
                labels = mini_batch[1]

                for image, label in zip(images, labels):
                    answer = np.zeros(10)
                    answer[label] = 1
                    self.backprop(image, answer)

                for j in range(1, len(self.layers)):
                    self.layers[j].bias_der = np.true_divide(self.layers[j].bias_der, mini_batch_size)
                    self.layers[j].weights_der = np.true_divide(self.layers[j].weights_der, mini_batch_size)
                    self.layers[j].bias -= self.layers[j].bias_der * eta
                    self.layers[j].weights -= self.layers[j].weights_der * eta + \
                                              np.true_divide(self.layers[j].weights * eta * lmbda, len(y))
                    self.layers[j].bias_der *= 0
                    self.layers[j].weights_der *= 0
        self.learn_progress[-1] = self.test(X_test, y_test)

    def test(self, X, y):
        rate = 0
        for image, label in zip(X, y):
            prediction = self.predict(image)
            if prediction == label:
                rate += 1
        return rate / len(X)

    def save(self, filename='network.json'):
        """ Saves trained network to file with path `filename` """
        weights_list = [[0] for i in range(len(self.layers) - 1)]
        bias_list = [[0] for i in range(len(self.layers) - 1)]

        for i in range(1, len(self.layers)):
            weights_list[i - 1] = self.layers[i].weights.tolist()
            bias_list[i - 1] = self.layers[i].bias.tolist()

        data = {"weights": weights_list,
                "bias": bias_list}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    @staticmethod
    def load(filename='network.json'):
        """ Loads trained network from file with path `filename` """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        net = Network([784, 100, 10])

        for i in range(1, len(net.layers)):
            net.layers[i].weights = np.array(data["weights"][i - 1])
            net.layers[i].bias = np.array(data["bias"][i - 1])

        return net


def main():
    X_train, X_test, y_train, y_test = load_mnist()
    net = Network.load('network2.json')
    print(net.test(X_test, y_test))
    # net = Network([784, 100, 10], smart_weights=True)
    # net.gradient_descent(X_train, y_train, X_test, y_test, epochs=10, lmbda=0)

    # net1 = Network([784, 100, 10], smart_weights=False)
    # net1.gradient_descent(X_train, y_train, X_test, y_test, epochs=10, lmbda=0)
    #
    # epochs = len(net.learn_progress)
    # plt.plot(range(1, epochs), net.learn_progress[1:epochs], label='Smart')
    # plt.plot(range(1, epochs), net1.learn_progress[1:epochs], label='Not smart')
    # plt.legend(title='Weights initialization')
    # plt.title('Weights initialization influence')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy on test set')
    # plt.show()


if __name__ == '__main__':
    main()
