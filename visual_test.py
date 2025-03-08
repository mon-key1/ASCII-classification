import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
from tensorflow.keras.models import load_model

model = load_model('mnist_model.keras')

canvas_width = 360
canvas_height = 360
image_size = (28, 28)

def preprocess_image(image):
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image = image.convert('L')
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 784)

def save_image_as_png(image, filename="output.png"):
    image.save(filename)

def update_prediction():
    canvas.postscript(file="tmp.ps", colormode='color')
    image = Image.open("tmp.ps")
    save_image_as_png(image, "output.png")
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)
    predicted_label = np.argmax(prediction)
    textPredict = ""
    if predicted_label == 1:
        textPredict = "A"
    elif predicted_label == 2:
        textPredict = "B"
    prediction_label.config(text=f"Predicted: {textPredict}")
    root.after(1000, update_prediction)

def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="Predicted: ")

root = tk.Tk()
root.title("Draw a digit")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

drawing = False
last_x, last_y = None, None
pen_width = 40

def start_drawing(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    if drawing:
        canvas.create_line(last_x, last_y, event.x, event.y, width=pen_width, capstyle=tk.ROUND, fill="black")
        last_x, last_y = event.x, event.y

def stop_drawing(event):
    global drawing
    drawing = False

canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

prediction_label = tk.Label(root, text="Predicted: ", font=("Arial", 24), fg="red")
prediction_label.pack()

root.after(50, update_prediction)

root.mainloop()
