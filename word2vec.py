import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec

class Wordtovec:
    def __init__(self):
        self.model = None
        self.model_words = None
        self.model_word_vectors = None

    def train_model(self, docs, queries):
        # Combine all sentences from docs and queries into a single list
        sentences = [sentence for doc in docs + queries for sentence in doc]
        
        # Initialize and train the Word2Vec model
        self.model = Word2Vec(sentences, window=3, min_count=0)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=30)
        
        # Save the trained model to a file
        self.model.save('model.bin')

    def load_model(self):
        # Load the trained Word2Vec model from the file
        self.model = Word2Vec.load('model.bin')
        
        # Extract the vocabulary and word vectors from the model
        self.model_words = list(self.model.wv.index_to_key)
        self.model_word_vectors = {word: self.model.wv[word] for word in self.model_words}

    def make_doc_vector(self, doc):
        # Initialize a vector to store the combined vector representation of the document
        combined_vector = np.zeros(100)
        
        # Iterate through each sentence in the document
        for sentence in doc:
            # Iterate through each word in the sentence
            for word in sentence:
                # Check if the word is present in the vocabulary
                if word in self.model_words:
                    # Add the word vector to the combined vector representation
                    combined_vector += self.model_word_vectors[word]
        
        # Normalize the combined vector representation
        combined_vector /= (norm(combined_vector) + 0.000001)
        return combined_vector

    def rank(self, docs, doc_ids, queries):

        # Initialize an instance of Wordtovec
        # word_to_vec = Wordtovec()

        # Train the Word2Vec model
        # word_to_vec.train_model(docs, queries)
        
        # Load the trained model
        self.load_model()
        print(len(self.model_words))
        # Initialize a list to store the ranked document IDs for each query
        ranked_doc_ids = []
        
        # Iterate through each query
        for query in queries:
            # Calculate the vector representation of the query
            query_vector = self.make_doc_vector(query)
            
            # Calculate the similarity between the query vector and each document vector
            similarities = [(np.dot(query_vector, self.make_doc_vector(doc)), i) for i, doc in enumerate(docs)]
            
            # Sort the similarities in descending order
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Extract the document IDs corresponding to the sorted similarities
            ranked_doc_ids.append([doc_ids[i] for _, i in similarities])
        print('done')
        return ranked_doc_ids
