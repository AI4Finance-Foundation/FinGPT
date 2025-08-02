import json
import cv2
import os
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
            cv2.putText(img, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

    def show_predictions(self, img):
        cv2.imshow('Predictions', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_predictions_to_json(self, output, json_path='output.json'):
        # Load the existing JSON if it exists
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        data["predictions"] = output
        return data

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
image_path = 'E:\\FinGPT-M\\Datasets\\Stock Charts for OCR\\UBER-1_png.rf.11c539b6bd5eed9f8bcd2cb6ad064427.jpg'

metadata_extractor = StockChartMetadataExtractor(image_path)
metadata_extractor.save_metadata_to_json("metadata.json")

stock_chart_predictor = StockChartTrendPredictor(model_path)
output, img = stock_chart_predictor.predict(image_path)
stock_chart_predictor.show_predictions(img)

combined_data = stock_chart_predictor.save_predictions_to_json(output, 'predictions.json')
combine_metadata_and_predictions('metadata.json', output, 'final_output.json')
file_path = ['metadata.json', 'predictions.json']
for file in file_path:
    if os.path.exists(file):
        os.remove(file)