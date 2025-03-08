
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
from tensorflow.keras.models import load_model

model = load_model('mnist_model.keras')

scale = 150
n = 3  # num of symbols
canvas_width = scale * n
canvas_height = scale

canvas_width_img = canvas_width / 1.4
canvas_height_img = canvas_height / 1.4

image_size = (28, 28)

def preprocess_image(image):
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image = image.convert('L')
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 784)

def save_image_as_png(image, filename="output.png"):
    image.save(filename)

def is_canvas_empty(canvas_image):
    image_array = np.array(canvas_image)
    return np.all(image_array == 255)

def update_prediction():
    canvas.postscript(file="tmp.ps", colormode='color')
    image = Image.open("tmp.ps")

    fullText = ""

    for j in range(0, n):
        p1 = canvas_height_img * j
        p2 = canvas_height_img * (j + 1)
        cur_image = image.crop((p1, 0, p2, canvas_height_img))
        save_image_as_png(cur_image, f"res/output{j}.png")

        if not is_canvas_empty(cur_image):
            cur_input_image = preprocess_image(cur_image)
            cur_prediction = model.predict(cur_input_image)
            cur_predicted_label = np.argmax(cur_prediction)

            textPredicted = chr(ord('A') + cur_predicted_label - 1) if cur_predicted_label != 0 else ""
            fullText += textPredicted

    prediction_label.config(text=f"Text: {fullText}")

def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="Text:")

root = tk.Tk()
root.title("Draw a digit")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

drawing = False
last_x, last_y = None, None
pen_width = 20

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
    update_prediction()  # Вызываем update_prediction при отпускании мыши

canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

prediction_label = tk.Label(root, text="Text: ", font=("Arial", 36), fg="red")
prediction_label.pack()

root.mainloop()
