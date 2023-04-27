from io import BytesIO
import os
from datasets import load_dataset
import requests
import concurrent.futures
import tqdm
from PIL import Image

output_folder = os.path.join(os.path.dirname(__file__), "aesthetics_65")
dataset = load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus")


urls = []
for item in tqdm.tqdm(dataset["train"]):
    urls.append(item["URL"])

finished = 0
max_items = len(urls)

def download_image(url, output_dir):
    global finished
    try:
        # Extract the image filename from the URL.
        filename = os.path.basename(url)
        # Create the full output path for the image.
        output_path = os.path.join(output_dir, filename)
        # Download the image using the requests library.
        response = requests.get(url)
        response.raise_for_status()
        # Save the image to the output path.
        with open(output_path, 'wb') as f:
            try:
                img = Image.open(BytesIO(response.content))
                img.save(f, format=img.format)
            except:
                print(f"Failed to open {url}")
                return
        finished += 1
        print("Finished {}/{}".format(finished, max_items))
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")

def parallel_download_images(urls, output_dir):
    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    # Use a ThreadPoolExecutor to download the images in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the download tasks to the executor.
        futures = [executor.submit(download_image, url, output_dir) for url in urls]
        # Wait for all the tasks to complete.
        concurrent.futures.wait(futures)

parallel_download_images(urls, output_folder)