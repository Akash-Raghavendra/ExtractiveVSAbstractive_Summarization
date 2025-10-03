import tkinter as tk
from tkinter import filedialog
from backend import load_and_summarize  

def browse_file():
    filename = filedialog.askopenfilename(title="Select a File", filetypes=[("Text files", "*.txt")])
    if filename:  
        file_path_entry.delete(0, tk.END) 
        file_path_entry.insert(0, filename)  

def reset():
    file_path_entry.delete(0, tk.END)  
    text_area.delete(1.0, tk.END)  

def exit_app():
    root.quit()  

def generate_summary():
    # Get the file path from the text entry
    file_path = file_path_entry.get()
    
    if file_path:
        try:
            # Call the summarization function from the backend
            abstractive_summary, extractive_summary, abstractive_rouge, extractive_rouge = load_and_summarize(file_path)
            
            # Display the summary in the text area
            text_area.delete(1.0, tk.END)  
            text_area.insert(tk.END, "Abstractive Summary:\n")
            text_area.insert(tk.END, abstractive_summary + "\n\n")
            text_area.insert(tk.END, "Extractive Summary:\n")
            text_area.insert(tk.END, extractive_summary + "\n\n")
            
            # Display ROUGE scores for trained model (abstractive) and extractive model
            text_area.insert(tk.END, "ROUGE Scores:\n")
            text_area.insert(tk.END, "Trained Model\n")
            text_area.insert(tk.END, f"ROUGE-1: {abstractive_rouge['rouge1'].fmeasure:.2f}\n")
            text_area.insert(tk.END, f"ROUGE-2: {abstractive_rouge['rouge2'].fmeasure:.2f}\n")
            text_area.insert(tk.END, f"ROUGE-L: {abstractive_rouge['rougeL'].fmeasure:.2f}\n")
            
            text_area.insert(tk.END, "\nExtractive Model:\n")
            text_area.insert(tk.END, f"ROUGE-1: {extractive_rouge['rouge1'].fmeasure:.2f}\n")
            text_area.insert(tk.END, f"ROUGE-2: {extractive_rouge['rouge2'].fmeasure:.2f}\n")
            text_area.insert(tk.END, f"ROUGE-L: {extractive_rouge['rougeL'].fmeasure:.2f}\n")
        
        except Exception as e:
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, f"Error: {str(e)}")
    else:
        text_area.delete(1.0, tk.END)  
        text_area.insert(tk.END, "No file selected!")

def output_as_file():
    output_filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if output_filename:
        with open(output_filename, 'w') as file:
            file.write(text_area.get(1.0, tk.END))  

root = tk.Tk()
root.title("Summarizer")

root.grid_rowconfigure(0, weight=0)  
root.grid_rowconfigure(1, weight=0)  
root.grid_rowconfigure(2, weight=0)  
root.grid_rowconfigure(3, weight=1)  
root.grid_rowconfigure(4, weight=0)  

root.grid_columnconfigure(0, weight=1)  
root.grid_columnconfigure(1, weight=2)  

header_label = tk.Label(root, text="Summarizer", font=("Helvetica", 16, "bold"))
header_label.grid(row=0, column=0, columnspan=3, pady=10)

reset_button = tk.Button(root, text="Reset", command=reset)
reset_button.grid(row=0, column=0, sticky="w", padx=10, pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.grid(row=0, column=2, sticky="e", padx=10, pady=10)

file_path_label = tk.Label(root, text="File Access Path:")
file_path_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
file_path_entry = tk.Entry(root, width=50)
file_path_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=1, column=2, padx=10, pady=5)

generate_button = tk.Button(root, text="Generate Summary", command=generate_summary)
generate_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

text_area = tk.Text(root, width=70, height=15)
text_area.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

output_button = tk.Button(root, text="Output as File", command=output_as_file)
output_button.grid(row=4, column=2, padx=10, pady=5, sticky="e")

root.mainloop()
