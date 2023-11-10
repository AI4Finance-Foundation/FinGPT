import pandas as pd
import easygui as gui

def find_abnormal_rows():
    # Prompt user to select CSV file
    file_path = gui.fileopenbox("Select CSV file", filetypes=["*.csv"])

    if file_path:
        # Read CSV file using pandas
        df = pd.read_csv(file_path, header=None)

        # Get the number of columns in the first row
        expected_num_columns = len(df.iloc[0])

        # Find rows with abnormal number of columns
        abnormal_rows = []
        for index, row in df.iterrows():
            if len(row) != expected_num_columns:
                abnormal_rows.append(index)

        # Report the abnormal row indexes
        if abnormal_rows:
            gui.msgbox("Abnormal rows found with inconsistent number of columns:\n{}".format(abnormal_rows))
        else:
            gui.msgbox("No abnormal rows found with inconsistent number of columns.")

    else:
        gui.msgbox("No file selected.")

if __name__ == '__main__':
    find_abnormal_rows()
