import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui import gui
from external_LLMs import external_LLMs
import pandas as pd
import openai
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from tqdm import tqdm

try:
    classification_choice = gui.ynbox("F1 calculation", "Do you want to classify the news using external LLMs?")

    if classification_choice:
        file_path = gui.fileopenbox("Select CSV file", "*.csv")
        df = pd.read_csv(file_path)
        column_names = df.columns.tolist()

        sentence_column = gui.buttonbox("Column Selection", "Select the column of sentence for classification:",
                                    column_names)
        if not sentence_column:
            raise ValueError("Invalid column selection")

    if not sentence_column:
        raise ValueError("Invalid column selection")


    gui.msgbox("Classification Complete")
except Exception as e:
    gui.exceptionbox(str(e))
    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    df.to_csv(output_file_path, index=False)