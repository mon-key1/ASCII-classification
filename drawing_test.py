import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
from tensorflow.keras.models import load_model
from typing import List, Any

model = load_model('models/mnist_model.keras')

def letters_extract(image_file: str, out_size=28, padding=10) -> List[Any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]

            letter_crop = cv2.copyMakeBorder(letter_crop, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            size_max = max(w + 2 * padding, h + 2 * padding)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - (h + 2 * padding) // 2
                letter_square[y_pos:y_pos + h + 2 * padding, 0:w + 2 * padding] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - (w + 2 * padding) // 2
                letter_square[0:h + 2 * padding, x_pos:x_pos + w + 2 * padding] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0], reverse=False)

    return letters

def preprocess_image(image):
    save_image_as_png(image, "res/out.png")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = image.convert('L')
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 784)

def save_image_as_png(image, filename="res/output.png"):
    image.save(filename)

def update_prediction():
    canvas.postscript(file="res/tmp.ps", colormode='color')
    image = Image.open("res/tmp.ps")
    save_image_as_png(image)
    
    letters = letters_extract("res/output.png")
    
    predicted_text = ""
    for letter in letters:
        x, w, letter_img = letter
        letter_img = Image.fromarray(letter_img)
        input_image = preprocess_image(letter_img)
        prediction = model.predict(input_image)
        predicted_label = np.argmax(prediction)
        predicted_text += chr(64 + predicted_label)
    
    prediction_label.config(text=f"Text: {predicted_text}")

    with open("res/drawing_output.txt", "w") as file:
        file.write(predicted_text + "\n")  

def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="Text: ")

root = tk.Tk()
root.title("Draw a digit")

canvas = tk.Canvas(root, width=900, height=270, bg="white")
canvas.pack()

drawing = False
last_x, last_y = None, None
pen_width = 5

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
    update_prediction()

canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

prediction_label = tk.Label(root, text="Text: ", font=("Arial", 36), fg="red")
prediction_label.pack()

root.mainloop()
