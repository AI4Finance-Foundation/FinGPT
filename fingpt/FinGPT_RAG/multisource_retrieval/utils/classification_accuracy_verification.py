import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui import gui
from external_LLMs import external_LLMs
import pandas as pd
import openai
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

try:
    classification_choice = gui.ynbox("F1 calculation", "Do you want to classify the news using external LLMs?")

    if classification_choice:
        file_path = gui.fileopenbox("Select CSV file", "*.csv")
        df = pd.read_csv(file_path)
        column_names = df.columns.tolist()

        actual_classifications_column = gui.buttonbox("F1 analysis", "Select the column of sentence with actual classifications:", column_names)
        if not actual_classifications_column:
            raise ValueError("Invalid column selection")


        predicted_classifications_column = gui.buttonbox("F1 analysis",
                                                      "Select the column of sentence with predicted classifications:",
                                                      column_names)
        if not predicted_classifications_column:
            raise ValueError("Invalid column selection")

    df = df.dropna(subset=[actual_classifications_column, predicted_classifications_column])
    df[actual_classifications_column] = df[actual_classifications_column].astype(int)
    df[predicted_classifications_column] = df[predicted_classifications_column].astype(int)
    computed_f1 = f1_score(df[actual_classifications_column], df[predicted_classifications_column], average=None)
    computed_accuracy_score = accuracy_score(df[actual_classifications_column], df[predicted_classifications_column])
    computed_precision_score = precision_score(df[actual_classifications_column], df[predicted_classifications_column], average=None)
    computed_recall_score = recall_score(df[actual_classifications_column], df[predicted_classifications_column], average=None)

    print("f1 score: ", computed_f1)
    print("accuracy score: ", computed_accuracy_score)
    print("precision score: ", computed_precision_score)
    print("recall score: ", computed_recall_score)

except Exception as e:
    gui.exceptionbox(str(e))