import os
import pandas as pd
import easygui as gui

def classify_csv_file():
    try:
        # Read CSV file
        file_path = gui.fileopenbox("Select CSV file", filetypes=["*.csv"])
        df = pd.read_csv(file_path)

        # Set "Seeking Alpha" as the value for rows under "classification" column from index 1 onwards
        df.loc[1:, "classification"] = "Seeking Alpha"

        # Save the classified CSV file
        base_name = os.path.basename(file_path)
        output_file_path = os.path.join(os.path.dirname(file_path), f"{os.path.splitext(base_name)[0]}_classified.csv")
        df.to_csv(output_file_path, index=False)

        gui.msgbox("Classification Complete")
    except Exception as e:
        gui.exceptionbox(str(e))

if __name__ == '__main__':
    classify_csv_file()
