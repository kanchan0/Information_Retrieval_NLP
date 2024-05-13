# Imports
import nltk
from nltk.corpus import stopwords

# Downloading resources
nltk.download('stopwords')

# getting nltk stopwords list
nltk_stopwords = set(stopwords.words('english'))

def custom_stopWords_list_generation(text,threshold_frequency):
    new_corpus = []
    for word in text:
        pattern = "1234567890!@#$%^&()[\"]./\,:'}{+-="
        word1 = ''
        for letter in word:
            if letter in pattern:
                word1 += ''
            else:
                word1 += letter
        new_corpus.append(word1)

    dictionary = {}
    for word in new_corpus:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    # print(dictionary)

    reversed_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    #print(list(reversed_dict.keys()))
    custom_stopWords = list(reversed_dict.keys())

    # Stopwords for Nltk library according to corpus
    stopwords_in_text = [word for word in custom_stopWords if word.lower() in nltk_stopwords]
    print("______________Stop words obtained NLTK on Cranfield DATA_________________")
    print(stopwords_in_text)
    print()
    print()

    custom_stopWords = custom_stopWords[:threshold_frequency]

    

    return custom_stopWords




def main():
    # path to tokenized doc. Specific my folder structure 
    tokenized_File_Path = './output/tokenized_docs.txt'
    
    # reading the file
    corpus = None
    with open(tokenized_File_Path,'r') as file:
        corpus = file.readline()
    corpus = corpus.split(' ')
    Custom_stopWords = custom_stopWords_list_generation(corpus,threshold_frequency=120)
    print("______________BOTTOMUP CUSTOM STOPWORDS on Cranfield DATA_________________")
    print(Custom_stopWords)


if __name__ == "__main__":
    main()