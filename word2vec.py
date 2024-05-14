import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec

class Wordtovec:

    def train_model(self, docs, queries):

        all_sentences = []

        for doc in docs:
            for sent in doc:
                all_sentences.append(sent)

        for query in queries:
            for sent in query:
                all_sentences.append(sent)

        model = Word2Vec(all_sentences, window=3, min_count=0)
        model.train(all_sentences, total_examples=model.corpus_count, epochs=30)

        model.save('model.bin')

    def make_doc_vec(self, doc):

        combined_vec = np.zeros(100)
        i = 0
        for sentence in doc:
            for word in sentence:
                if(word in self.model_words):
                    if(i == 0):
                        combined_vec = self.model_wv_dict[word]
                        i += 1
                    else:
                        combined_vec = combined_vec +  self.model_wv_dict[word]

        combined_vec = combined_vec / (norm(combined_vec) + 0.000001) #Adding 0.000001 is to avoid division by zero in case
        return combined_vec

    def rank(self, docs, doc_ids, queries):

        self.train_model(docs, queries)
        self.model = Word2Vec.load('model.bin')
        self.model_words = list(self.model.wv.index_to_key) #loads all the vocab words
        self.model_wv_dict = {word : self.model.wv[word] for word in self.model_words}  #creates a dict for all words in vocab and maps it with its vec

        docIDs_oredered = []
        for q in queries:
            all_comparisions = []
            q_vec = self.make_doc_vec(q)  #creating the sum of all word embeddings vector in query
            similarities = []
            i = 0
            for doc in docs:
                doc_vec = self.make_doc_vec(doc)  #creating the sum of all word embeddings vector in docs
                similarity = np.dot(q_vec, doc_vec) / ((norm(q_vec) * norm(doc_vec)) + 0.00001)
                similarities.append((similarity, i))
                i += 1
            similarities.sort(key=lambda x : x[0], reverse=True)  #sorting a list of similarities in descending order based on the first element of each tuple in the list

            for _, i in similarities:
                all_comparisions.append(doc_ids[i])
            docIDs_oredered.append(all_comparisions)

        return docIDs_oredered