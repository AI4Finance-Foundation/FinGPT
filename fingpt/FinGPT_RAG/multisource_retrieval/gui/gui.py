import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def ynbox(title, message):
    response = messagebox.askyesno(title, message)
    return response

def fileopenbox(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title=title, filetypes=[(filetypes, filetypes)])
    root.destroy()
    return filepath

def buttonbox(title, message, choices):
    choice = None

    def on_choice(c):
        nonlocal choice
        choice = c
        root.quit()

    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=message).pack()
    for c in choices:
        tk.Button(root, text=c, command=lambda c=c: on_choice(c)).pack()
    root.mainloop()
    root.destroy()

    return choice

def enterbox(message, title, default):
    return simpledialog.askstring(title, message, initialvalue=default)

def msgbox(message):
    messagebox.showinfo("Information", message)

def exceptionbox(message):
    messagebox.showerror("Error", message)