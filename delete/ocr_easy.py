import pandas as pd
import easyocr
import os

root_dir = "dataset"
# Create output directory
os.makedirs(fr'{root_dir}//ocrtext_easy', exist_ok=True)

# Initialize EasyOCR with GPU support
# gpu=True enables GPU acceleration
# ['en'] sets English language (add more languages as needed, e.g., ['en', 'hi'])
print("Initializing EasyOCR (this may take a moment on first run)...")
reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR initialized successfully!\n")


def reconstruct_text_structure(ocr_result):
    """
    Reconstruct text with preserved structure (newlines and spaces)
    based on the bounding box positions
    """
    if not ocr_result:
        return ""

    # Extract all text boxes with their positions
    boxes_data = []
    for detection in ocr_result:
        box = detection[0]  # Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = detection[1]  # Extracted text
        confidence = detection[2]  # Confidence score

        # Calculate center y-coordinate and left x-coordinate
        y_coords = [point[1] for point in box]
        x_coords = [point[0] for point in box]
        y_center = sum(y_coords) / len(y_coords)
        x_left = min(x_coords)
        x_right = max(x_coords)

        boxes_data.append({
            'text': text,
            'y': y_center,
            'x': x_left,
            'x_right': x_right,
            'confidence': confidence
        })

    # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
    boxes_data.sort(key=lambda b: (b['y'], b['x']))

    # Group text by lines based on y-coordinate proximity
    lines = []
    current_line = []
    y_threshold = 15  # Pixels tolerance for same line

    for box in boxes_data:
        if not current_line:
            current_line.append(box)
        else:
            # Check if this box is on the same line as previous
            if abs(box['y'] - current_line[0]['y']) < y_threshold:
                current_line.append(box)
            else:
                # Sort current line by x-coordinate and add to lines
                current_line.sort(key=lambda b: b['x'])
                lines.append(current_line)
                current_line = [box]

    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda b: b['x'])
        lines.append(current_line)

    # Reconstruct text with proper spacing
    reconstructed_text = []
    for line in lines:
        line_text = []

        for i, box in enumerate(line):
            if i > 0:
                # Calculate horizontal gap between current and previous word
                prev_box = line[i - 1]
                gap = box['x'] - prev_box['x_right']

                # Add space if gap is significant (more than 10 pixels)
                if gap > 10:
                    line_text.append(' ')

            line_text.append(box['text'])

        reconstructed_text.append(''.join(line_text))

    return '\n'.join(reconstructed_text)


def process_images():
    """Process all images using EasyOCR on GPU"""
    # Read the CSV file
    df = pd.read_csv(fr"{root_dir}//labels.csv")
    print(f"Starting EasyOCR processing with GPU acceleration...")
    print(f"Total images to process: {len(df)}\n")

    success_count = 0
    errors = []

    # Process each image
    for idx, row in df.iterrows():
        image_name = row.iloc[0]
        image_path = os.path.join(fr"{root_dir}//images", image_name)
        output_file = os.path.join(fr"{root_dir}//ocrtext_easy", image_name.removesuffix('.png') + '.txt')
        if os.path.exists(output_file):
            success_count += 1
            print(f"\rProcessed {idx + 1}/{len(df)} - Success: {success_count}", end="")
            continue

        try:
            # Check if file exists
            if not os.path.exists(image_path):
                error_msg = f"Warning: File not found - {image_path}"
                print(error_msg)
                errors.append((idx, error_msg))
                continue

            # Perform OCR on the image
            # paragraph=False gives us individual text boxes which we'll reconstruct
            result = reader.readtext(image_path, paragraph=False)

            # Reconstruct text with preserved structure
            text = reconstruct_text_structure(result)

            # Save to file
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(text)

            success_count += 1
            print(f"\rProcessed {idx + 1}/{len(df)} - Success: {success_count}", end="")

        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(f"\n{error_msg}")
            errors.append((idx, error_msg))

    # Print summary
    print(f"\n\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(df)} images")

    if errors:
        print(f"\nTotal errors: {len(errors)}")
        print("\nErrors encountered:")
        for idx, msg in errors[:10]:  # Show first 10 errors
            print(f"  Image {idx}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    process_images()