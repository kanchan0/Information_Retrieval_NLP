import re
import numpy as np
from itertools import chain
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from nltk.stem import PorterStemmer
#np.seterr(divide='ignore', invalid='ignore')

class LSA:

    # constructor
    def __init__(self) -> None:
        self.index = None
        self.stemmer = PorterStemmer()
        self.vocab = {}
    

    '''
    This method performs Singular Value Decomposition(SVD) on tf matrix. It extracts the left 
    singular vectors (U), singular values (sig), and right singular vectors (Vt). It removes
    the singular values to keep only the top k singular values. The resulting matrices U, V, and
    diagonal matrix S are stored in US, VS, and S respectively.
    '''
    def SVD_decomposition(self):
        U, sig, Vt = svds(self.tf_matrix, k=700, which='LM')
        V = Vt.T
        t = 1e-10
        req_sing_vals = np.diag(np.array([i if i > t else 0.0 for i in sig]))
        self.S = np.diag(sig)
        self.VS = np.dot(V, self.S)
        self.US = np.dot(U, req_sing_vals)
    
    def rank(self, docs, docIDs, queries):
        self.build_vocab(docs)
        self.build_index(docs)
        self.SVD_decomposition()
        
        docIDs_oredered = []
        for query in queries:
            all_comparisions = []
            query_vec = [0] * len(self.vocab)
            words_of_q = re.compile(r'\W+').split(query)
            for words in words_of_q:
                stemmed_word = self.stemmer.stem(words)
                if stemmed_word in self.vocab:
                    query_vec[self.vocab[stemmed_word]] += 1
            
            reduced_query = np.dot(self.US.T, query_vec)
            norm_query = norm(reduced_query)

            similarities = []
            for i, row in enumerate(self.VS):
                similarity = np.dot(row, reduced_query) / (norm(row) * norm_query)
                similarities.append((similarity, i))
            similarities.sort(key=lambda x : x[0], reverse=True)

            for _, i in similarities:
                all_comparisions.append(docIDs[i])
            docIDs_oredered.append(all_comparisions)
			
        return docIDs_oredered
    
    def build_index(self, docs):
        self.matrix_row = []
        self.matrix_col = []
        self.matrix_data = []
        for i in range(len(docs)):
            self.index = {}
            for words in list(chain.from_iterable(docs[i])):
                stemmed_word = self.stemmer.stem(words)
                if stemmed_word not in self.index.keys():
                    self.index[stemmed_word] = 1
                else:
                    self.index[stemmed_word] += 1
            
            for (t, f) in self.index.items():
                self.matrix_row.append(self.vocab[t])
                self.matrix_col.append(i)
                self.matrix_data.append(f)
        self.tf_matrix = csc_matrix((self.matrix_data, (self.matrix_row, self.matrix_col)), shape=(self.no_of_words, len(docs))).astype(float)

    def build_vocab(self, docs):
        self.no_of_words = 0
        for i in range(len(docs)):
            for words in list(chain.from_iterable(docs[i])):
                stemmed_word = self.stemmer.stem(words)
                if stemmed_word not in self.vocab.keys():
                    self.vocab[stemmed_word] = self.no_of_words
                    self.no_of_words += 1

        

    
