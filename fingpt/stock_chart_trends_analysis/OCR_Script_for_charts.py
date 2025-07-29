import easyocr
import cv2
import matplotlib.pyplot as plt
from ocr_utilities import parse_chart_metadata

# General-purpose function to extract metadata from any image source
def extract_image_metadata(image_source):
    """Extract metadata from an image source (path or image object)."""
    # Load image
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise ValueError(f"Unable to load image from path: {image_source}")
        img_path = image_source
    else:
        img = image_source
        img_path = None

    reader = easyocr.Reader(['en'], verbose=False)
    # OCR on full image
    results = reader.readtext(img_path if img_path else img)
    # OCR on cropped and resized axis region
    height = img.shape[0]
    cropped = img[int(height * 0.85):, :]
    resized = cv2.resize(cropped, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    results += reader.readtext(resized)

    # Visualization (optional)
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("OCR Results on Cropped & Resized X-Axis Region")
    plt.show()

    print("\n--- OCR Results ---")
    for text, conf in results:
        print(f"{text} ({conf:.2f})")

    raw_texts = [text for _, text, _ in results]
    metadata = parse_chart_metadata(raw_texts)
    print(metadata)
    return metadata

# Example usage:
metadata = extract_image_metadata(r"C:\\Users\\rahul\\OneDrive\\Desktop\\FinGPT-M\\Datasets\\Stock Charts for OCR\\Intuit-1_png.rf.1100f8c84c6b26930087532b40e842b5.jpg")