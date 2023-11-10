import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui import gui
from external_LLMs import external_LLMs
import pandas as pd

# Classify sentences
try:
    classification_choice = gui.ynbox("Classification", "Do you want to classify the news?")

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
    default_classification_prompt = ". For financial statement above, determine its origin (based on your existing knowledge). Your answer should be either \"Twitter\" or \"Seeking Alpha\" or \"Reuters\" or \"WSJ\""
    classification_prompt = gui.enterbox("Modify the classification prompt:", "Custom Classification Prompt",
                                         default_classification_prompt)

    if not classification_prompt:
        classification_prompt = default_classification_prompt

    for row_index, row in df.iloc[1:].iterrows():
        target_sentence = row[sentence_column]
        classification_response = external_LLMs.extract_classification(target_sentence, classification_prompt)
        if "Twitter" in classification_response:
            classification_response = "Twitter"
        elif "Seeking Alpha" in classification_response:
            classification_response = "Seeking Alpha"
        elif "Reuters" in classification_response:
            classification_response = "Reuters"
        elif "WSJ" in classification_response:
            classification_response = "WSJ"
        else:
            classification_response = "Unknown"
        df.at[row_index, "classification"] = classification_response  # Assign classification response to the new column

    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    df.to_csv(output_file_path, index=False)
    gui.msgbox("Classification Complete")
except Exception as e:
    gui.exceptionbox(str(e))
    print("Error occurred at row index:", row_index)
    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    df.to_csv(output_file_path, index=False)