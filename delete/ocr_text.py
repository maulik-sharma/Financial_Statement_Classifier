import pandas as pd
import pytesseract
from PIL import Image
import os
from multiprocessing import Pool, cpu_count
from functools import partial

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mauli\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Create output directory
os.makedirs('dataset//ocrtext', exist_ok=True)


def process_image(row_data, base_path="dataset//images", output_path="dataset//ocrtext"):
    """Process a single image with OCR"""
    idx, image_name = row_data
    image_path = os.path.join(base_path, image_name)

    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return (idx, f"Warning: File not found - {image_path}", False)

        # Open image and extract text
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        text = text.strip()

        # Save to file
        output_file = os.path.join(output_path, image_name.removesuffix('.png') + '.txt')
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text)

        return (idx, None, True)

    except Exception as e:
        return (idx, f"Error processing {image_path}: {str(e)}", False)


def main():
    # Read the CSV file
    df = pd.read_csv("dataset//labels.csv")

    # Prepare data for multiprocessing (index and image name)
    image_data = [(idx, row.iloc[0]) for idx, row in df.iterrows()]

    # Use 4 processes (or fewer if less CPU cores available)
    num_processes = min(8, cpu_count())
    print(f"Starting OCR processing with {num_processes} processes...")

    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_image, image_data)

    # Report results
    success_count = sum(1 for _, _, success in results if success)
    print(f"\n\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(df)} images")

    # Print any errors
    errors = [(idx, msg) for idx, msg, success in results if not success and msg]
    if errors:
        print("\nErrors encountered:")
        for idx, msg in errors:
            print(f"  Image {idx}: {msg}")


if __name__ == "__main__":
    main()