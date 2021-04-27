Romance Language Learner by Ali Abusulb and Carson Hamel

To run the code:
1. Install the dependencies with "pip install nltk" and "pip install googletrans==3.1.0a0" (this is the alpha version due to some
errors with the current regular version)
2. Run the main.py file with "python main.py"

This will start a loop where you are asked to input text to classify. Simple imput one or more words in one of the target 
languages (French, Italian, Spanish or Portuguese) and it will attempt to classify it. Input 'q' at any time to exit

Then the program will output the accuracy of the language classifier, on a test dataset.

The next part of the program may take some time to run. It will first calculate the accuracy of the cognate classification
function based on a small hand-labelled dataset we created. It will print this number to the screen.

It will then calculate the lexical overlap between each language with the others using a list of nouns. 
Lexical overlap is the percent of words among 2 languages with shared meanings that are cognates. The program will show this percentage
for each of the languages in relation to all of the others.

