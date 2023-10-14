import pandas as pd
import easygui as gui

def find_different_rows():
    # Prompt user to select CSV file
    file_path = gui.fileopenbox("Select CSV file", filetypes=["*.csv"])

    if file_path:
        # Read CSV file using pandas
        df = pd.read_csv(file_path)

        # Ensure "link" column exists
        if "link" not in df.columns:
            gui.msgbox("'link' column is missing.")
            return

        # Find rows where "link" does not contain "http"
        non_http_rows = df[~df['link'].str.contains("http", na=False)]

        # Report the count of non-http links
        if not non_http_rows.empty:
            gui.msgbox("Total number of rows without 'http' in 'link' column: {}".format(len(non_http_rows)))
        else:
            gui.msgbox("No rows found without 'http' in 'link' column.")

    if file_path:
        # Read CSV file using pandas
        df = pd.read_csv(file_path)

        # Ensure "text" and "contextualized sentences" columns exist
        if "text" not in df.columns or "contextualized_sentence" not in df.columns:
            gui.msgbox("Either or both 'text' and 'contextualized_sentences' columns are missing.")
            return

        # Find rows where "text" and "contextualized sentences" values are different
        different_rows = df[df['text'] != df['contextualized_sentence']]

        # Report the different row indexes
        if not different_rows.empty:
            gui.msgbox("total number is {}".format(len(different_rows.index.tolist())))
        else:
            gui.msgbox("No rows found with different values for 'text' and 'contextualized_sentences'.")

    else:
        gui.msgbox("No file selected.")

if __name__ == '__main__':
    find_different_rows()
