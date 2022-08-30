import network2
import numpy as np
from PIL import Image, ImageDraw
import PIL
from tkinter import *
import matplotlib.pyplot as plt
import functools

net = network2.Network.load('network2.json')


# App for creating and examining own examples for MNIST handwritten digits recognition problem

def trunc(values, decs=0):
    """ Leaves `decs` digits after . in real numbers(`values`) """
    return np.trunc(values * 10 ** decs) / (10 ** decs)


# tkinter app parameters
width, height = 400, 400
center = height // 2
background_color = 'white'
drawing_color = 'black'
drawing_density = 10
canvas_padding = 100


def submit():
    """ Converts drawn image to 784 dim numpy array and prints out network's decision"""
    data = drawing.convert('L')
    data = data.resize((28, 28), PIL.Image.ANTIALIAS)
    data = np.array(data)
    data = np.true_divide(data, 255)
    data *= -1
    data += 1
    data = trunc(data, 2)
    res = net.feedforward(data.reshape(28 ** 2))
    for i in range(len(res)):
        if res[i] >= 0.1:
            print(f"{i}:        {np.round(res[i], 3)} \n")
    print("----------------------------------- \n \n")
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()


def paint(event, color):
    """ Draws or erases image """
    x1, y1 = (event.x - drawing_density), (event.y - drawing_density)
    x2, y2 = (event.x + drawing_density), (event.y + drawing_density)
    cv.create_rectangle(x1, y1, x2, y2, fill=color, width=0)
    draw.rectangle([x1, y1, x2, y2], fill=color, width=0)


def delete_all(event):
    """ Erase all canvas contents """
    cv.delete("all")
    draw.rectangle([0, 0, width, height], fill=background_color)


# this app doesn't use direct drawing from tkinter canvas as it was hard(not possible?) to implement
# instead it draws simultaneously on PIL image(later fed to network) and tkinter canvas(for a user to see)
drawing = PIL.Image.new('RGB', (width, height), background_color)
draw = ImageDraw.Draw(drawing)

root = Tk()

cv = Canvas(root, width=width, height=height, bg=background_color)
cv.pack(expand=YES, fill=BOTH, padx=(0, canvas_padding))
cv.bind('<B1-Motion>', functools.partial(paint, color=drawing_color))
cv.bind('<B3-Motion>', functools.partial(paint, color=background_color))
cv.bind('<Button-2>', delete_all)

button = Button(text='submit', command=submit)
button.pack()
root.mainloop()
