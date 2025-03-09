from PIL import Image, ImageDraw, ImageFont, ImageTransform
import os
import random
from data import get_dataset_test
from data import get_dataset_npz

def generate(chars,fonts,output_folder,augment_count):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_size = (28, 28)
    
    for font_file in os.listdir(fonts_folder):
        if font_file.endswith('.otf') or font_file.endswith('.ttf'):
            font_path = os.path.join(fonts_folder, font_file)

            try:
                font = ImageFont.truetype(font_path, size=20)
            except Exception as e:
                print(f"Error loading font {font_file}: {e}")
                continue

            for char in chars:
                try:
                    image = Image.new('L', image_size, color=255)
                    draw = ImageDraw.Draw(image)

                    bbox = draw.textbbox((0, 0), char, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
                    draw.text(position, char, font=font, fill=0)

                    output_filename = f"{char}_{os.path.splitext(font_file)[0]}_original.png"
                    output_path = os.path.join(output_folder, output_filename)
                    image.save(output_path)

                    for i in range(augment_count):
                        transformed_image = image.copy()
                        draw = ImageDraw.Draw(transformed_image)

                        transformed_image = Image.eval(transformed_image, lambda x: 255 - x)

                        if i % 2 == 0:
                            scale_x = random.uniform(0.9, 1.3)
                            transformed_image = transformed_image.transform(
                                image_size,
                                ImageTransform.AffineTransform((scale_x, 0, 0, 0, 1, 0)))
                        else:
                            scale_y = random.uniform(0.9, 1.3)
                            transformed_image = transformed_image.transform(
                                image_size,
                                ImageTransform.AffineTransform((1, 0, 0, 0, scale_y, 0)))

                        angle = random.uniform(-30, 30)
                        transformed_image = transformed_image.rotate(angle, resample=Image.BICUBIC, expand=True)

                        transformed_image = Image.eval(transformed_image, lambda x: 255 - x)
                        transformed_image = transformed_image.resize((28, 28))

                        output_filename = f"{char}_{os.path.splitext(font_file)[0]}_variation_{i+1}.png"
                        output_path = os.path.join(output_folder, output_filename)
                        transformed_image.save(output_path)

                except Exception as e:
                    print(f"Error creating image for text {char} with font {font_file}: {e}")

# add symbols that should be in your dataset
chars = ['A','B','C','D','E','F','G','H','I','J','K','L','N','O','P','R','S','T','U','V','Y','Z']

# you can also customize fonts (make sure the symbols are displayed correctly with each font)
fonts_folder = 'data/fonts-otf'
output_folder = 'data/dataset/train'

# copies with stretching and rotation for each (170) symbol instanse
augment_count = 10

# dataset size = fonts count * chars count * augment_count
# default = 170 * 26 * 10

generate(chars,fonts_folder,output_folder,augment_count)
get_dataset_test.generate(chars,fonts_folder,'data/dataset/test')
get_dataset_npz.get_npz()
