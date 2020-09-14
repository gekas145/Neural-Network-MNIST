import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import math


class Network:

    class InputLayer:
        def __init__(self, input):
            self.input = input


    class MaxPool:
        def __init__(self, maps_num):
            self.masks = [0 for j in range(maps_num)]
            self.output = np.zeros(maps_num * 144)
            self.delta = np.zeros(maps_num * 144)
            for i in range(maps_num):
                self.masks[i] = np.zeros((24, 24))


    class ConvolutionalLayer:
        def __init__(self, maps_num):
            self.maps = [0 for j in range(maps_num)]
            self.kernels = [0 for j in range(maps_num)]
            self.bias = np.zeros(maps_num)
            self.delta = [0 for j in range(maps_num)]
            self.weights_der = [0 for j in range(maps_num)]
            self.bias_der = [0 for j in range(maps_num)]

            for i in range(maps_num):
                self.maps[i] = np.zeros((24, 24))
                self.kernels[i] = np.random.randn(5, 5)
                self.bias[i] = np.random.randn(1)[0]
                self.bias_der[i] = 0
                self.weights_der[i] = np.zeros((5, 5))
                self.delta[i] = np.zeros((24, 24))




    class LastLayer:
        def __init__(self):
            self.activations = np.zeros(10)
            self.delta = np.zeros(10)
            self.weights = np.zeros(3)  # will be modified while Network creation
            self.bias = np.zeros(10)
            self.weights_der = np.zeros(3)
            self.bias_der = np.zeros(10)


    def __init__(self, maps_num):
        self.imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
        self.lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
        self.testimage = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/t10k-images.idx3-ubyte')
        self.testlable = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/t10k-labels.idx1-ubyte')
        self.imagearray = np.true_divide(self.imagearray, 255)
        self.testimage = np.true_divide(self.testimage, 255)
        self.validation_data_image = self.imagearray  # for express net testing
        self.validation_data_lable = self.lablesarray
        self.input_layer = Network.InputLayer(np.zeros(3))  # will be updated when calling feedforward function
        self.convolutional_layer = Network.ConvolutionalLayer(maps_num)
        self.maxpool = Network.MaxPool(maps_num)
        self.last_layer = Network.LastLayer()
        self.last_layer.weights = np.random.randn(10, maps_num * 144)
        self.last_layer.weights_der = np.zeros((10, maps_num * 144))



    def sigmoid(self, z):
        """ The sigmoid function """
        return 1 / (1 + np.exp(-z))


    def update_mini_batch(self, n, m):
        return self.validation_data_image[n: n + m], self.validation_data_lable[n: n + m]



    def feedforward(self, image):
        self.input_layer.input = image

        for k in range(len(self.convolutional_layer.maps)):
            for i in range(24):
                for j in range(24):
                    self.convolutional_layer.maps[k][i][j] += \
                        np.dot(self.input_layer.input[i:i+5, j:j+5].reshape(25), self.convolutional_layer.kernels[k].reshape(25)) + self.convolutional_layer.bias[k]
            self.convolutional_layer.maps[k] = self.sigmoid(self.convolutional_layer.maps[k])




        output_pos = 0  # remembers position of last output insertion
        for k in range(len(self.convolutional_layer.maps)):
            for i in range(0, 23, 2):
                for j in range(0, 23, 2):
                    slice = self.convolutional_layer.maps[k][i:i+2, j:j+2]
                    m, n = np.unravel_index(slice.argmax(), slice.shape)
                    self.maxpool.masks[k][m+i][n+j] = 1
            output = self.maxpool.masks[k] * self.convolutional_layer.maps[k]
            # a = self.maxpool.masks[k].reshape(24*24)
            # a = a[a != 0]
            # if len(a) != 144:
            #     print(self.maxpool.masks[k])
            #     return 0
            output = output.reshape(len(output)**2)
            output = output[output != 0]

            # print(len(output))
            self.maxpool.output[output_pos:output_pos + len(output)] = output
            output_pos += len(output)

        z = np.dot(self.last_layer.weights, self.maxpool.output) + self.last_layer.bias
        self.last_layer.activations = self.sigmoid(z)

        for j in range(len(self.maxpool.masks)):
            self.maxpool.masks[j] = np.zeros((24, 24))

        return self.last_layer.activations


    def backprop(self, input, output):
        res = self.feedforward(input)
        self.last_layer.delta = output - res
        self.last_layer.bias_der += self.last_layer.delta
        for i in range(len(self.last_layer.delta)):
            self.last_layer.weights_der[i] += self.maxpool.output * self.last_layer.delta[i]

        self.maxpool.delta = np.dot(np.transpose(self.last_layer.weights), self.last_layer.delta) * (self.maxpool.output * (1 - self.maxpool.output))

        delta_pos = 0  # to divide maxpool.delta array into delta arrays for different feature maps
        for i in range(len(self.convolutional_layer.maps)):
            self.convolutional_layer.delta[i] = self.maxpool.masks[i].reshape(len(self.maxpool.masks[i])**2)
            non_zero_pos = np.where(self.convolutional_layer.delta[i] != 0)[0]
            for j in range(len(non_zero_pos)):
                self.convolutional_layer.delta[i][non_zero_pos[j]] = self.maxpool.delta[delta_pos + j]
            delta_pos += len(non_zero_pos)
            self.convolutional_layer.delta[i] = self.convolutional_layer.delta[i].reshape(len(self.maxpool.masks[i]), len(self.maxpool.masks[i]))

        # below weight and bias derivatives of convolutional layer are counted

        for k in range(len(self.convolutional_layer.maps)):
            for n in range(5):
                for m in range(5):
                    for i in range(24):
                        for j in range(24):
                            self.convolutional_layer.weights_der[k][n][m] += self.input_layer.input[i+n][j+m] * self.convolutional_layer.delta[k][i][j]
                            self.convolutional_layer.bias_der[k] += self.convolutional_layer.delta[k][i][j]
                            self.convolutional_layer.bias_der[k] += self.convolutional_layer.delta[k][i][j]

    def gradient_descent(self, eta = 1, m = 10, epochs = 30):
        n = 0  # number of current mini-batch
        k = int(len(self.validation_data_image) / (m * epochs))  # quantity of mini-batches in one epoch
        rate = 0
        for i in range(epochs):


            for t in range(k):
                # tmp = self.update_mini_batch(n, m)
                # n += m
                images = self.validation_data_image[n:n + m]
                lables = self.validation_data_lable[n:n + m]

                n += m
                # print(lables)
                # print(lables)




                for image, label in zip(images, lables):
                    y = np.zeros(10)
                    y[label] = 1
                    self.backprop(image, y)


                for j in range(len(self.convolutional_layer.maps)):
                    self.last_layer.bias_der[j] = np.true_divide(self.last_layer.bias_der[j], m)
                    self.last_layer.weights_der[j] = np.true_divide(self.last_layer.weights_der[j], m)
                    self.convolutional_layer.weights_der[j] = np.true_divide(self.convolutional_layer.weights_der[j], m)
                    self.convolutional_layer.bias_der[j] = np.true_divide(self.convolutional_layer.bias_der[j], m)

                    self.last_layer.bias[j] -= self.last_layer.bias_der[j] * eta
                    self.last_layer.weights[j] -= self.last_layer.weights_der[j] * eta
                    self.convolutional_layer.bias[j] -= self.convolutional_layer.bias_der[j] * eta
                    self.convolutional_layer.kernels[j] -= self.convolutional_layer.weights_der[j] * eta

                    self.last_layer.bias[j] *= 0
                    self.last_layer.weights[j] *= 0
                    self.convolutional_layer.bias[j] *= 0
                    self.convolutional_layer.kernels[j] *= 0



        for j in range(len(self.testimage)):
            ans = self.feedforward(self.testimage[j])
            if np.amax(ans) == ans[self.testlable[j]]:
                rate += 1
        print(rate)











net = Network(3)

image = net.imagearray[0]
k = 0
# for i in net.feedforward(image):
#     print(f"{k} : {i}")
#     k += 1

# plt.imshow(image, cmap=plt.cm.binary)
# plt.show()
# net.backprop(image, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
# plt.imshow(net.convolutional_layer.maps[0], cmap=plt.cm.binary)
# plt.show()
net.gradient_descent()
# print(net.convolutional_layer.weights_der[0])


# a = np.random.randn(24, 24)
# a = net.sigmoid(a)
# plt.imshow(a, cmap=plt.cm.binary)
# plt.show()






