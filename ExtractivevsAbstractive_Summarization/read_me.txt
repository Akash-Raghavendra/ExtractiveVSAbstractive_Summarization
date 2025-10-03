PLEASE READ THIS FILE TO UNDERSTAND FILE STRUCTURE AND A GUIDE ON HOW TO EXECUTE THIS PROJECT

Scripts/Files and their purpose - 
1) summarization_model and logs - The model was trained on google collab. This is the model downloaded after training and it's logs. 
2) frontend.py - This script contains code for the GUI.
3) backend.py - This script enables the functioning of the buttons in the frontend.
4) model.py - This script contains the code for the model which was trained on google collab.
5) requirements.txt - This text file contains the python libraries needed for this project.
6) samsum-train.csv, samsum-test.csv, samsum-validation.csv - These csv files contain the samsum dataset in three parts (train, test, validation) which was required for training the model.
7) text.txt and text2.txt - These text files contain input which can be used to generate summary for. 


How to run this project - 
1) If the neccessary libraries to run this project are not installed in your machine, under this directory, open terminal and run the command 'pip install -r requirements.txt'.
2) Have text that you want to summarize ready in the form of a text file. For convenience it can be placed in the same project folder. For the purpose of running this project I have already placed two files (text.txt and text2.txt) which can be used as input. 
3) Next run the frontend.py script. This should open up the frontend GUI.
4) Use the file browser option to select the text file which has the text to be summarized.
5) Once the text file has been selected, click on the generate summary button to see the generated summaries, and their corresponding rogue scores under in the text area. 
6) If the output is wanted in a text file, click the output as file button in the bottom right corner of the GUI window and you can follow the instructions in the correspoding window to do so. 