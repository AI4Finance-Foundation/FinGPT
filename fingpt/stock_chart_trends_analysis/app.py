import os
import sys
sys.path.insert(0, r"/content/FinGPT-M/fingpt/stock_chart_trends_analysis")
from config import YOLO_WEIGHTS
from ui.gradio_app import create_app

if __name__ == "__main__":
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    
    demo = create_app()
    demo.launch(share = True, debug = True)