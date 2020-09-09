import numpy as np
import matplotlib.pyplot as plt
import idx2numpy


class Network:
    class MaxPool:
        def __init__(self, maps_num, dim):
            self.masks = np.zeros(maps_num)
            self.maps = np.zeros(maps_num)
            for i in range(maps_num):
                self.maps[i] = np.zeros(dim, dim)
                self.masks[i] = np.zeros(dim, dim)


    class ConvolutionalLayer:
        def __init__(self, maps_num, dim, kernel_dim):
            self.maps = np.zeros(maps_num)
            self.kernels = np.zeros(maps_num)
            for i in range(maps_num):
                self.maps[i] = np.zeros(dim, dim)
                self.kernels[i] = np.random.randn((kernel_dim, kernel_dim))



    class LastLayer:
        def __init__(self):
            self.activations = np.zeros(10)
            self.delta = np.zeros(10)
            self.weights = np.zeros(3)  # will be modified while Network creation
            self.bias = np.zeros(10)


    def __init__(self, maps_num, dims, kernel_dim):
        self.imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
        self.lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
        self.imagearray = np.true_divide(self.imagearray, 255)
        self.testimage = np.true_divide(self.testimage, 255)
        self.validation_data_image = self.imagearray[0:100]  # for express net testing
        self.validation_data_lable = self.lablesarray[0:100]
        self.layers = [1, 1, 1, 1]
        self.layers[0] = Network.ConvolutionalLayer(1, 28, 5)
        self.layers[1] = Network.ConvolutionalLayer(maps_num[0], dims[0], kernel_dim)
        self.layers[2] = Network.MaxPool(maps_num[1], dims[1])
        self.layers[3] = Network.LastLayer()
        self.layers[3].weights = np.random.randn(10, dims[1]*dims[1]*maps_num[1])


    def feedforward(self, image):
        pass







