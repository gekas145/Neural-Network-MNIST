
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

# def trunc(values, decs=0):
#     return np.trunc(values*10**decs)/(10**decs)
#
#
# width, height = 200, 200
# center = height//2
# white = (255, 255, 255)
# green = (0,128,0)


# tkinter app parameters above


# def save():
    # image1.save('C:/Users/yevhe/Downloads/file.png')
    # data = image1.convert('L')
    # data = data.resize((28, 28), PIL.Image.ANTIALIAS)
    # data = np.array(data)
    # data = np.true_divide(data, 255)
    # data *= -1
    # data += 1
    # data = trunc(data, 1)
    # res = net.feedforward(load_data(data))
    # for i in range(len(res)):
    #     if res[i] >= 0.1:
    #         print(f"{i}:        {np.round(res[i], 3)} \n")
    # print("----------------------------------- \n \n")
    # plt.imshow(data, cmap=plt.cm.binary)
    # plt.show()

# def paint(event):
    # python_green = "#476042"
    # x1, y1 = (event.x - 8), (event.y - 8)
    # x2, y2 = (event.x + 8), (event.y + 8)
    # cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    # draw.line([x1, y1, x2, y2],fill="black",width=5)

# above functions are used for self networking testing




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
                self.layers[i].set_weights(np.random.randn(layers_sizes[i], layers_sizes[i-1]))
                self.layers[i].set_weights_der(np.zeros((layers_sizes[i], layers_sizes[i-1])))



    def sigmoid(self, z):
        """ The sigmoid function """
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """  Derivative of the sigmoid function """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def load_data(self, x):
        """ Reshapes matrix which represents an image into single 1-D array """
        return x.reshape(784)


    def feedforward(self, input):
        """  Analyses input and returns so called decision """
        self.layers[0].set_activations(input)
        for i in range(1, len(self.layers)):
            z = np.dot(self.layers[i].weights, self.layers[i-1].activations) + self.layers[i].bias
            self.layers[i].set_z(z)
            z = self.sigmoid(z)
            self.layers[i].set_activations(z)

        return self.layers[-1].activations



    def update_mini_batch(self, n, m):
        return self.imagearray[n: n + m], self.lablesarray[n: n + m]



    def backprop(self, input, output):
        res = self.feedforward(input)
        self.layers[-1].set_error((res - output) * self.sigmoid_prime(self.layers[-1].z))
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].change_bias_der(self.layers[i].error)
            for j in range(len(self.layers[i].error)):
                self.layers[i].change_weights_der(self.layers[i-1].activations * self.layers[i].error[j], j)
            if i == 1:
                break
            self.layers[i-1].set_error(np.dot(np.transpose(self.layers[i].weights), self.layers[i].error) * self.sigmoid_prime(self.layers[i-1].z))





    def gradient_descent(self, eta = 3, m = 10, epochs = 30):
        n = 0  # number of current mini-batch
        k = int(len(self.imagearray)/(m * epochs))  # quantity of mini-batches in one epoch
        self.learn_progress = np.zeros(epochs)
        for i in range(epochs):

            for j in range(len(self.testimage)):
                w = np.zeros(10)  # right answer
                w[self.testlable[j]] = 1
                self.learn_progress[i] += np.sum(np.power(self.feedforward(self.load_data(self.testimage[j])) - w, 2))
            self.learn_progress[i] /= len(self.testimage)

            for t in range(k):
                tmp = self.update_mini_batch(n, m)
                n += m
                images = tmp[0]
                lables = tmp[1]

                for image, label in zip(images, lables):
                    y = np.zeros(10)
                    y[label] = 1
                    self.backprop(self.load_data(image), y)

                for j in range(1, len(self.layers)):
                    self.layers[j].bias_der = np.true_divide(self.layers[j].bias_der, m)
                    self.layers[j].weights_der = np.true_divide(self.layers[j].weights_der, m)
                    self.layers[j].bias -= self.layers[j].bias_der * eta
                    self.layers[j].weights -= self.layers[j].weights_der * eta
                    self.layers[j].bias_der *= 0
                    self.layers[j].weights_der *= 0

        self.test()
        # print(self.feedforward(load_data(self.testimage[11])))
        # data = Image.open('C:/Users/yevhe/Downloads/test.png').convert('L')
        # data = data.resize((28, 28), Image.ANTIALIAS)
        # data = np.array(data)
        # data = np.true_divide(data, 255)
        # data *= -1
        # data += 1
        # print(self.feedforward(load_data(data)))
        # plt.imshow(self.testimage[11], cmap=plt.cm.binary)
        # plt.show()
        # for i in range(10):
        #     print(self.testlable[i])
        #     print(self.feedforward(load_data(self.testimage[i])))
        #     print("----------------------")






    def test(self):
        rate = 0
        for image, label in zip(self.testimage, self.testlable):
            # y = np.zeros(10)
            # y[label] = 1
            # rate += np.sum(np.power(self.feedforward(load_data(image)) - y, 2))
            # if np.sum(np.power(self.feedforward(load_data(image)) - y, 2)) < 0.2:
            #     rate += 1
            ans = self.feedforward(self.load_data(image))
            if np.amax(ans) == ans[label]:
                rate += 1
        print(f"rate is {rate}/{len(self.testlable)}")
        # print(f"Mean square error: {rate/len(self.testlable)}")














net = Network([784, 30, 10])
net.gradient_descent()

y = net.learn_progress
x = np.arange(len(y))
plt.title("Network learning process")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.plot(x,y)
plt.show()

# for i in range(10):
#     plt.imshow(net.testimage[i], cmap=plt.cm.binary)
#     plt.show()


# while True:
#     root = Tk()
#
#     cv = Canvas(root, width=width, height=height, bg='white')
#     cv.pack()
#
#
#     # memory only, not visible
#     image1 = PIL.Image.new("RGB", (width, height), white)
#     draw = ImageDraw.Draw(image1)
#
#     cv.pack(expand=YES, fill=BOTH)
#     cv.bind("<B1-Motion>", paint)
#
#     button = Button(text="save", command=save)
#     button.pack()
#     root.mainloop()

# plt.imshow(net.testimage[10], cmap=plt.cm.binary)
# plt.show()



# imagearray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-images.idx3-ubyte')
# lablesarray = idx2numpy.convert_from_file('C:/Users/yevhe/Downloads/samples/train-labels.idx1-ubyte')
# imagearray = imagearray / 255
# for i in range(5):
#     print(lablesarray[i])
#
# for i in imagearray[0]:
#     print(i)
#
# plt.imshow(net.testimage[106], cmap=plt.cm.binary)
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

