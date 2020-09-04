import network2

net = network2.Network([784, 100, 10])
net.gradient_descent()

from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *


# App for creating and examining own examples for MNIST handwritten digits recognition problem

def trunc(values, decs=0):
    """ Leaves `decs` digits after . in real numbers(`values`) """
    return np.trunc(values*10**decs)/(10**decs)


width, height = 200, 200
center = height//2
white = (255, 255, 255)
green = (0,128,0)


# tkinter app parameters above


def save():
    """ Converts drawn image to 784 dim numpy array and prints out network's decision"""
    data = image1.convert('L')
    data = data.resize((28, 28), PIL.Image.ANTIALIAS)
    data = np.array(data)
    data = np.true_divide(data, 255)
    data *= -1
    data += 1
    data = trunc(data, 1)
    res = net.feedforward(net.load_data(data))
    for i in range(len(res)):
        if res[i] >= 0.1:
            print(f"{i}:        {np.round(res[i], 3)} \n")
    print("----------------------------------- \n \n")
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()

def paint(event):
    """ Draws image """
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)



while True:
    root = Tk()

    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    button = Button(text="save", command=save)
    button.pack()
    root.mainloop()