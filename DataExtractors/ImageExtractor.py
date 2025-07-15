import fitz

def extract_images_from_pdf(pdf_path : str):
    doc = fitz.open(pdf_path) 
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            base_image = doc.extract_image(images[img_index][0])
            image_bytes = base_image["image"]
            with open(f"extracted_images/page{page_index}_{img_index}.png", "wb") as f:
                f.write(image_bytes)
