import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def find_index_of_sentence_with_max_common_words(sentence, sentence_list):
    max_common_words = 0
    index_with_max_common_words = 0

    # Iterate over each sentence in the list
    for i, s in enumerate(sentence_list):
        # Split the sentences into words
        words_sentence = set(sentence.lower().split())
        words_s = set(s.lower().split())
        
        # Find the number of common words between the sentence and s
        common_words_count = len(words_sentence.intersection(words_s))

        # Update the maximum common words and the index of the sentence with maximum common words
        if common_words_count > max_common_words:
            max_common_words = common_words_count
            index_with_max_common_words = i

    return index_with_max_common_words


def getResponse(ints, intents_json,msg):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            inx = find_index_of_sentence_with_max_common_words(msg, i['patterns'])
            result = i['responses'][inx]
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents,msg)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

# Create the main window
base = Tk()
base.title("HR CHATBOT")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="black", height="8", width="50", font=("Arial", 12))

# Configure additional options for the ChatLog widget
ChatLog.config(state=DISABLED, wrap="word")

# Set a custom color for the ChatLog widget's background
ChatLog.config(bg="#f2f2f2")

# Set a custom color for the ChatLog widget's text
ChatLog.config(fg="#333333")

# Set a custom font for the ChatLog widget
ChatLog.config(font=("Arial", 12))

# Add a border to the ChatLog widget
ChatLog.config(borderwidth=2, relief="solid")

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(
    base,
    text="Send",
    font=("Arial", 12, 'bold'),
    width=10,
    height=2,
    bd=10,
    bg="#0000ff",
    activebackground="#0000ff",
    fg='#ffffff',
    command=send
)

# Apply hover effect
SendButton.bind("<Enter>", lambda e: SendButton.config(bg="#3c9d9b"))
SendButton.bind("<Leave>", lambda e: SendButton.config(bg="#32de97"))

# Add a drop shadow effect
SendButton.config(
    highlightcolor="#000000",
    highlightthickness=1,
    highlightbackground="#000000",
    bd=0
)

# Position the button using grid layout
SendButton.grid(row=1, column=0, pady=10, padx=10, sticky="w")

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="30", height="3", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=310)
SendButton.place(x=326, y=401, height=90, width=68)

base.mainloop()