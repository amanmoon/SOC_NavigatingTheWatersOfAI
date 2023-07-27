import nltk
import sys
import os 
import math

FILE_MATCHES = 7
SENTENCE_MATCHES = 2

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    path = os.path.join(os.getcwd(), directory)
    dict_of_file = {}
    for file_name in os.listdir(path):
        with open(os.path.join(path, file_name), 'r', encoding='utf-8') as file:
            dict_of_file[file_name] = file.read()
    return dict_of_file

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document=document.lower()
    stopword = set(nltk.corpus.stopwords.words('english'))
    word_list= nltk.tokenize.word_tokenize(document)
    wordlist=[word for word in word_list if word.isalpha() and word not in stopword]        
         
  
    return wordlist


# files = load_files("corpus")    
# file_words = {filename: tokenize(files[filename]) for filename in files}  #dict filename:tokenised list
def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_idf={}
    for txtfile in documents:        
        for words in documents[txtfile]:
            if words in word_idf:
                word_idf[words]+=1
            else:
                word_idf[words]=1
    for word in word_idf:
        word_doc_freq=0
        for textfile in documents:
            if word in documents[textfile]:
                word_doc_freq+=1
        word_idf[word]=math.log(len(documents.keys())/word_doc_freq)
    return word_idf
                    
# idf=compute_idfs(file_words)
# idf=sorted(idf.items(), key=lambda item: item[1])


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf = {}
    for txtfile in files:
        for word in query:
            if word in files[txtfile]:
                if word in tfidf:
                    tfidf[txtfile] += idfs[word]
                else:
                    tfidf[txtfile] = idfs[word]

    sorted_files = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
    top_n_files = [filename for filename, score in sorted_files][:n]

    return top_n_files

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    def sentence_score(sentence):
        total_idf = sum(idfs[word] for word in sentence if word in query)
        query_term_density = sum(1 for word in sentence if word in query) / len(sentence)
        return total_idf, query_term_density

    sorted_sentences = sorted(sentences.items(), key=lambda item: sentence_score(item[1]), reverse=True)
    top_n_sentences = [sentence for sentence, _ in sorted_sentences][:n]

    return top_n_sentences



if __name__ == "__main__":
    main()
