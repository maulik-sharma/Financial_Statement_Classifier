import easyocr
import os
import requests
import base64
import time

SERVER_URL = os.getenv("url","https://ocr.cpscan.com")
gpu = True if os.getenv("device","gpu") == "gpu" else False

# Configuration
GET_WORK_ENDPOINT = f"{SERVER_URL}/get_work"
SUBMIT_WORK_ENDPOINT = f"{SERVER_URL}/submit_work"

# Initialize EasyOCR with GPU support
print("Initializing EasyOCR (this may take a moment on first run)...")
reader = easyocr.Reader(['en'], gpu=gpu)
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


def get_work():
    """Request work from the server"""
    try:
        response = requests.get(GET_WORK_ENDPOINT, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "status" in data:
            return None, None

        return data["file_name"], data["image"]
    except requests.RequestException as e:
        print(f"Error requesting work: {e}")
        return None, None


def submit_work(filename, text):
    """Submit completed work to the server"""
    try:
        payload = {
            "filename": filename,
            "text": text
        }
        response = requests.post(SUBMIT_WORK_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, dict) and result.get("status") == "success":
            return True
        elif result is True:
            return True
        else:
            print(f"Server returned error for {filename}: {result}")
            return False
    except requests.RequestException as e:
        print(f"Error submitting work for {filename}: {e}")
        return False


def process_image_from_base64(base64_image):
    """Process a base64 encoded image with EasyOCR"""
    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(base64_image)
        # image = Image.open(BytesIO(image_bytes))

        # Convert PIL Image to format EasyOCR can use
        # EasyOCR can work with PIL images directly
        result = reader.readtext(image_bytes, paragraph=False)

        # Reconstruct text with preserved structure
        text = reconstruct_text_structure(result)

        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def main():
    """Main client loop"""
    print(f"Starting OCR client...")
    print(f"Connecting to server at: {SERVER_URL}\n")

    success_count = 0
    error_count = 0
    no_work_count = 0

    while True:
        # Request work from server
        filename, base64_image = get_work()

        if filename is None:
            no_work_count += 1
            if no_work_count == 1:
                print("\nNo work available from server.")

            if no_work_count >= 3:
                print("No work available after 3 attempts. Exiting.")
                break

            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
            continue

        # Reset no work counter when we get work
        no_work_count = 0

        print(f"Processing: {filename}")

        # Process the image
        text = process_image_from_base64(base64_image)

        if text is None:
            print(f"Failed to process {filename}")
            error_count += 1
            # Submit with None to let server reassign it
            submit_work(filename, None)
            continue

        # Submit the result
        if submit_work(filename, text):
            success_count += 1
            print(f"✓ Successfully completed {filename} (Total: {success_count})")
        else:
            error_count += 1
            print(f"✗ Failed to submit {filename}")

        # Small delay to prevent overwhelming the server
        time.sleep(0.5)

    # Print final summary
    print(f"\n{'=' * 50}")
    print(f"Client session complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nClient stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")