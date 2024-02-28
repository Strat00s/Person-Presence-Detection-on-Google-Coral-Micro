import requests
import time
from PIL import Image
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
            text_data, image_data = parse_data(data)

            # Save text data
            text_file_path = os.path.join(folder_name, f"{text_counter}.txt")
            with open(text_file_path, "w") as text_file:
                text_file.write(text_data)
            text_counter += 1

            # Save image
            image_file_path = os.path.join(folder_name, f"{image_counter}.jpeg")
            save_image(image_data, image_file_path)
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
    print(text_decoder)

    # Extract the text data
    text_data = splits[0]  + "\n"
    text_data += splits[1] + "\n"
    text_data += splits[2] + "\n"
    text_data += splits[3] + "\n"
    text_data += splits[4] + "\n"
    bbox_cnt = splits[4]
    bboxes_len = len(splits[5])
    bboxes = splits[5].strip(";").split(";")
    for i in range(0, int(bbox_cnt)):
        text_data += ",".join(bboxes[:6]) + "\n"

    # Process image data (skipping actual conversion logic for brevity)
    # Assuming image data starts after the CSV data
    csv_len = sum(len(split) for split in splits[:5]) + 6 + bboxes_len # Commas
    print(csv_len)
    image_data = data[csv_len:]
    print(len(image_data))

    return text_data, image_data

def save_image(image_data, filename):
    image_width = 320
    image_height = 320

    # Assuming image_data is in RGB format, convert it to an image
    img_array = np.frombuffer(image_data, dtype=np.uint8).reshape((image_height, image_width, 3))
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename)
# Replace 'your_uri_here' with your actual URI path
uri = "http://192.168.1.112/camera_stream"
fetch_and_save_data(uri)
