import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
from tensorflow.keras.models import load_model

model = load_model('mnist_model.keras')

canvas_width = 720
canvas_height = 360
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

    left_image = image.crop((0, 0, 270, 270))
    right_image = image.crop((270, 0, 540, 270))
    full_image = image.crop((0, 0, 520, 270))

    left_predicted_label = 0
    right_predicted_label = 0

    if not is_canvas_empty(left_image):
        left_input_image = preprocess_image(left_image)
        left_prediction = model.predict(left_input_image)
        left_predicted_label = np.argmax(left_prediction)
    
    if not is_canvas_empty(right_image):
        right_input_image = preprocess_image(right_image)
        right_prediction = model.predict(right_input_image)
        right_predicted_label = np.argmax(right_prediction)

    save_image_as_png(left_image, "r/output1.png")
    save_image_as_png(right_image, "r/output2.png")
    save_image_as_png(full_image, "r/output3.png")
    
    textPredictLeft = chr(ord('A') + left_predicted_label - 1) if left_predicted_label != 0 else ""
    textPredictRight = chr(ord('A') + right_predicted_label - 1) if right_predicted_label != 0 else ""
    
    prediction_label.config(text=f"Text: {textPredictLeft + textPredictRight}")
    root.after(3000, update_prediction)

def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="Text:")

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

prediction_label = tk.Label(root, text="Text: ", font=("Arial", 36), fg="red")
prediction_label.pack()

root.after(3000, update_prediction)

root.mainloop()
