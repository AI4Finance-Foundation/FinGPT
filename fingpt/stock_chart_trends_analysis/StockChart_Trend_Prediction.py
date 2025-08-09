import json
import cv2
import os
import datetime
from shapely import box
from ultralytics import YOLO
from OCR_Script_for_charts import StockChartMetadataExtractor

class StockChartTrendPredictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model_path = model_path

    def predict(self, img_path):
        results = self.model(img_path)
        boxes = results[0].boxes
        img = cv2.imread(img_path)
        output = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            p1 = (int(xyxy[0]), int(xyxy[1]))
            p2 = (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            output.append({
                "class": class_name,
                "confidence": conf,
                "bounding_box": {
                    "x_min": xyxy[0],
                    "y_min": xyxy[1],
                    "x_max": xyxy[2],
                    "y_max": xyxy[3]
                }
            })

        return output, img

    def estimate_price_from_y(self, y, image_height, price_min, price_max):
        if price_min is None or price_max is None:
            return "N/A"
        price = price_max - (y / image_height) * (price_max - price_min)
        return round(price, 2)

    def show_predictions(self, img):
        cv2.imshow('Predictions', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_predictions_to_json(self, output, json_path='output.json', img=None, metadata=None):
        h, w, _ = img.shape

        # Prepare all time values from metadata
        all_times = []
        for session in metadata.get("sessions", []):
            all_times.extend(session["time"])
        all_times_sorted = sorted(
            set(all_times), key=lambda x: datetime.datetime.strptime(x, "%I:%M %p")
        )

        price_min, price_max = metadata.get("price_range", [None, None])
        predictions_with_ranges = []

        for pred in output:
            box = pred["bounding_box"]
            x_min, x_max = box["x_min"], box["x_max"]
            y_start = box["y_min"]
            y_end = box["y_max"]

            # Estimate time range
            left_idx = int((x_min / w) * len(all_times_sorted)) if all_times_sorted else 0
            right_idx = int((x_max / w) * len(all_times_sorted)) if all_times_sorted else 0
            left_idx = max(0, min(len(all_times_sorted) - 1, left_idx))
            right_idx = max(0, min(len(all_times_sorted) - 1, right_idx))

            time_range = [
                all_times_sorted[left_idx] if all_times_sorted else "N/A",
                all_times_sorted[right_idx] if all_times_sorted else "N/A"
            ]

            # Estimate prices
            start_price = self.estimate_price_from_y(y_start, h, price_min, price_max)
            end_price = self.estimate_price_from_y(y_end, h, price_min, price_max)

            updated_pred = {
                "class": pred["class"],
                "confidence": pred["confidence"],
                "bounding_box": box,
                "time_range": time_range,
                "price_range": {
                    "start_price": start_price,
                    "end_price": end_price
                }
            }

            predictions_with_ranges.append(updated_pred)

        # Merge with metadata
        metadata["predictions"] = predictions_with_ranges

        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Final output saved to {json_path}")
        return metadata

def combine_metadata_and_predictions(metadata_path, predictions_output, final_output_path):
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Combine metadata with predictions
    combined_data = metadata
    combined_data["predictions"] = predictions_output

    # Save the combined result to a JSON file
    with open(final_output_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined data saved to {final_output_path}")

model_path = 'E:\\FinGPT-M\\fingpt\\stock_chart_trends_analysis\\best.pt'
image_path = 'E:\\FinGPT-M\\Datasets\\Stock Charts for OCR\\ICICIBANK-1_png.rf.41ec497ec8e11a58f8a0df076f4cda48.jpg'

metadata_extractor = StockChartMetadataExtractor(image_path)
metadata = metadata_extractor.extract_metadata()

predictor = StockChartTrendPredictor(model_path)
output, img = predictor.predict(image_path)

predictor.show_predictions(img)

predictor.save_predictions_to_json(output, 'final_output.json', img, metadata)
file_path = ['metadata.json', 'predictions.json']
for file in file_path:
    if os.path.exists(file):
        os.remove(file)