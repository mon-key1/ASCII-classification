from PIL import Image, ImageDraw, ImageFont, ImageTransform
import os
import random

def generate(chars,fonts,output_folder,augment_count=1):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_size = (28, 28)
    
    for font_file in os.listdir('data/fonts-otf'):
        if font_file.endswith('.otf') or font_file.endswith('.ttf'):
            font_path = os.path.join(fonts, font_file)

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
                    os.remove(os.path.join(output_folder,f"{char}_{os.path.splitext(font_file)[0]}_original.png"))

                except Exception as e:
                    print(f"Error creating image for text {char} with font {font_file}: {e}")

