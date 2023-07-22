import os
import nltk
def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    path=os.getcwd()
    path=os.path.join(path,directory)
    dict_of_file={file_name : open(os.path.join(path,file_name),'r').read() for file_name in os.listdir(path)}    
    return dict_of_file

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document.lower()
    word_list= nltk.tokenize.word_tokenize(document)
    for word in word_list:
        if word.isalpha():
            word_list.remove(word)
    return word_list


files = load_files("corpus")
file_words = {filename: tokenize(files[filename]) for filename in files}