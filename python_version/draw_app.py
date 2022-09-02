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
canvas_padding = 200
prediction_rectangle_height = 5
prediction_rectangle_left_padding = 10
prediction_rectangle_top_padding = 50
prediction_rectangle_gap = 20
prediction_rectangle_max_length = 50
prediction_rectangle_label_distance = 10


def analyze():
    """ Converts drawn image to 784 dim numpy array and prints out network's decision"""
    data = drawing.convert('L')
    data = data.resize((28, 28), PIL.Image.ANTIALIAS)
    data = np.array(data)
    data = np.true_divide(data, 255)
    data *= -1
    data += 1
    data = trunc(data, 2)
    res = net.feedforward(data.reshape(28 ** 2))
    update_prediction_canvas(res)


def paint(event, color):
    """ Draws or erases image """
    x1, y1 = (event.x - drawing_density), (event.y - drawing_density)
    x2, y2 = (event.x + drawing_density), (event.y + drawing_density)
    cv.create_rectangle(x1, y1, x2, y2, fill=color, width=0)
    draw.rectangle([x1, y1, x2, y2], fill=color, width=0)
    analyze()


def delete_all(event):
    """ Erase all canvas contents """
    cv.delete('all')
    draw.rectangle([0, 0, width, height], fill=background_color)


def update_prediction_canvas(res):
    cv1.delete('all')
    cv1.create_text(canvas_padding // 2, 10, text='Network prediction')
    for i in range(len(res)):
        cv1.create_text(rectangles[i][0] - prediction_rectangle_label_distance,
                        rectangles[i][1], text=str(i) + ': ')
        cv1.create_rectangle(rectangles[i][0],
                             rectangles[i][1],
                             rectangles[i][0] + prediction_rectangle_max_length * res[i],
                             rectangles[i][1] + prediction_rectangle_height,
                             fill=drawing_color)


# this app doesn't use direct drawing from tkinter canvas as it was hard(not possible?) to implement
# instead it draws simultaneously on PIL image(later fed to network) and tkinter canvas(for a user to see)
drawing = PIL.Image.new('RGB', (width, height), background_color)
draw = ImageDraw.Draw(drawing)

root = Tk()

cv = Canvas(root, width=width, height=height, bg=background_color)
cv.pack(expand=YES, fill=BOTH, side='left')
cv.bind('<B1-Motion>', functools.partial(paint, color=drawing_color))
cv.bind('<B3-Motion>', functools.partial(paint, color=background_color))
cv.bind('<Button-2>', delete_all)
cv1 = Canvas(root, width=canvas_padding, height=height, bg=background_color)
cv1.pack(expand=YES, fill=BOTH, side='right')

rectangles = []
for i in range(10):
    rectangles.append((prediction_rectangle_left_padding + prediction_rectangle_label_distance,
                       i * prediction_rectangle_gap + prediction_rectangle_top_padding +
                       prediction_rectangle_label_distance))

update_prediction_canvas(np.zeros(10))

root.mainloop()
