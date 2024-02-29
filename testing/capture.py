import requests
import time
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import os


def create_batch_folder():
    base_folder = "batch_"
    counter = 1
    while True:
        folder_name = f"{base_folder}{counter}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        counter += 1

def fetch_and_save_data(uri, interval=0.1):
    folder_name = create_batch_folder()
    image_counter = 1
    text_counter = 1

    while True:
        try:
            # Fetch data
            response = requests.get(uri)
            response.raise_for_status()
            data = response.content

            # Assuming the format is consistent with the JavaScript example
            text_data, image_data, bboxes = parse_data(data)

            # Save text data
            text_file_path = os.path.join(folder_name, f"{text_counter}.txt")
            with open(text_file_path, "w") as text_file:
                text_file.write(text_data)
            text_counter += 1

            # Save image
            image_file_path = os.path.join(folder_name, f"{image_counter}")
            save_image(image_data, image_file_path, bboxes)
            image_counter += 1

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        time.sleep(interval)

def decode_as_much_as_possible(binary_data):
    text_parts = []
    for i in range(len(binary_data)):
        try:
            # Try to decode the current slice
            part = binary_data[:i+1].decode('utf-8')
            text_parts.append(part[-1])  # Only keep the last character to avoid duplication
        except UnicodeDecodeError:
            # Stop at the first undecodable byte
            break
    return ''.join(text_parts)

def parse_data(data):
    # Convert the binary data to text for parsing
    text_decoder = decode_as_much_as_possible(data)
    splits = text_decoder.split(",")

    # Extract the text data
    text_data = splits[0]  + "\n"
    text_data += splits[1] + "\n"
    text_data += splits[2] + "\n"
    text_data += splits[3] + "\n"
    text_data += splits[4] + "\n"
    bbox_cnt = splits[4]
    bboxes_len = len(splits[5])
    bboxes = splits[5].strip(";").split(";")
    bboxes_fin = []
    for i in range(0, int(bbox_cnt)):
        text_data += ",".join(bboxes[:6]) + "\n"
        bboxes_fin.append([int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3]), int(bboxes[4]), int(bboxes[5])])

    # Process image data (skipping actual conversion logic for brevity)
    # Assuming image data starts after the CSV data
    csv_len = sum(len(split) for split in splits[:5]) + 6 + bboxes_len # Commas
    print(csv_len)
    image_data = data[csv_len:]
    print(len(image_data))

    return text_data, image_data, bboxes_fin

def save_image(image_data, filename, bboxes):
    image_width = 320
    image_height = 320

    # Assuming image_data is in RGB format, convert it to an image
    img_array = np.frombuffer(image_data, dtype=np.uint8).reshape((image_height, image_width, 3))
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename + "_raw.jpeg")

    # Create a drawing context to add a rectangle
    draw = ImageDraw.Draw(img)
    # Draw the rectangle with red outline and specified width
    for bbox in bboxes:
        print(bbox)
        color = "red"
        if bbox[1] == 1:
            color = "red"
        if bbox[1] == 2:
            color = "blue"
        if bbox[1] == 3:
            color = "yellow"
        if bbox[1] == 4:
            color = "green"
        draw.rectangle([bbox[4], bbox[5], bbox[2], bbox[3]], outline=color, width=2)
        font_path = "/usr/share/fonts/truetype/firacode/FiraCode-Bold.ttf"
        font = ImageFont.truetype(font=font_path, size=20)
        draw.text([bbox[4], bbox[5] - 20], f"{bbox[0]}", fill=color, font=font)
    
    # Save the modified image with the rectangle
    modified_filename = f"{filename}_res.jpeg"
    img.save(modified_filename)

# Replace 'your_uri_here' with your actual URI path
uri = "http://192.168.2.2/camera_stream"
fetch_and_save_data(uri)
