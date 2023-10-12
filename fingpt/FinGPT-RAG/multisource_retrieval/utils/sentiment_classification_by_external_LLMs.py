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

    df["openai_inferred_sentiment_from_RAG"] = ""  # Create a new column named "classification"
    default_classification_prompt = ". For financial statement above, determine its sentiment (based on your existing knowledge). Your answer should be either \"negative\" or \"neutral\" or \"positive\""
    classification_prompt = gui.enterbox("Modify the classification prompt:", "Custom Classification Prompt",
                                         default_classification_prompt)

    if not classification_prompt:
        classification_prompt = default_classification_prompt

    counter = 0
    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    for row_index, row in df.iloc[1:].iterrows():
        target_sentence = row[sentence_column]
        classification_response = external_LLMs.extract_classification(target_sentence, classification_prompt)
        if "negative" in classification_response:
            classification_response = 0
        elif "positive" in classification_response:
            classification_response = 1
        elif "neutral" in classification_response:
            classification_response = 2
        df.at[row_index, "openai_inferred_sentiment_from_RAG"] = classification_response

        counter += 1

        # Save the DataFrame to a CSV file every 10 rows
        if counter % 10 == 0:
            df.to_csv(output_file_path, index=False)

    gui.msgbox("Classification Complete")
except Exception as e:
    gui.exceptionbox(str(e))
    print("Error occurred at row index:", row_index)
    output_file_path = os.path.splitext(file_path)[0] + "_classified.csv"
    df.to_csv(output_file_path, index=False)