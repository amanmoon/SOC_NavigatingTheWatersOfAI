import os
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
directory=input("directory:")
path = os.path.join(os.getcwd(), directory)
dict_of_file = {}
for file_name in os.listdir(path):
    with open(os.path.join(path, file_name), 'r', encoding='utf-8') as file:
        contents=file.read()
        contents=contents.split(" ")
        stopwords=set(nltk.corpus.stopwords.words("english"))
        content=[word for word in contents if (word not in stopwords and word.isalpha())] 
        print(len(content))