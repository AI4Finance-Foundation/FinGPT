import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui import gui
from external_LLMs import external_LLMs
import pandas as pd

# Classify sentences
try:
    classification_choice = gui.ynbox("Sentiment Classification", "Do you want to classify the news using external LLMs?")

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

    df["classification"] = ""  # Create a new column named "classification"
    default_classification_prompt = ". For financial statement above, determine its sentiment (based on your existing knowledge). Your answer should be either \"negative\" or \"neutral\" or \"positive\""
    classification_prompt = gui.enterbox("Modify the classification prompt:", "Custom Classification Prompt",
                                         default_classification_prompt)

    if not classification_prompt:
        classification_prompt = default_classification_prompt

    for row_index, row in df.iloc[1:].iterrows():
        target_sentence = row[sentence_column]
        classification_response = external_LLMs.extract_classification(target_sentence, classification_prompt)
        df.at[row_index, "classification"] = classification_response  # Assign classification response to the new column

    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    df.to_csv(output_file_path, index=False)
    gui.msgbox("Classification Complete")
except Exception as e:
    gui.exceptionbox(str(e))
    print("Error occurred at row index:", row_index)
    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    df.to_csv(output_file_path, index=False)