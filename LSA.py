import numpy as np
from scipy.sparse import csc_matrix
from nltk.stem import PorterStemmer
from numpy.linalg import norm
from scipy.sparse.linalg import svds
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class LSA:
    def __init__(self) -> None:
        self.index = None
        self.stemmer = PorterStemmer()
        self.vocab = {}
        self.tfs = None  # The term frequency matrix
        self.no_of_words = 0

    def preprocess_docs(self, docs):
        """
        Preprocess the documents by removing stop words and stemming the remaining words.
        """
        preprocessed_docs = []
        stop_words = set(stopwords.words('english'))
        for doc in docs:
            preprocessed_doc = []
            for sent in doc:
                filtered_words = [word for word in sent if word.isalnum() and word.lower() not in stop_words]
                stemmed_sent = [self.stemmer.stem(word) for word in filtered_words]
                preprocessed_doc.append(stemmed_sent)
            preprocessed_docs.append(preprocessed_doc)
        return preprocessed_docs

    def build_vocab(self, docs):
        """
        Build the vocabulary from the preprocessed documents.
        """
        for doc in docs:
            for sent in doc:
                for word in sent:
                    if word not in self.vocab:
                        self.vocab[word] = self.no_of_words
                        self.no_of_words += 1

    def build_index(self, docs):
        """
        Build the term frequency matrix from the preprocessed documents.
        """
        matrix_row, matrix_col, matrix_data = [], [], []
        for i, doc in enumerate(docs):
            index = {}
            for word in chain.from_iterable(doc):
                if word not in index:
                    index[word] = 1
                else:
                    index[word] += 1

            for t, f in index.items():
                matrix_row.append(self.vocab[t])
                matrix_col.append(i)
                matrix_data.append(f)

        self.tfs = csc_matrix((matrix_data, (matrix_row, matrix_col)), shape=(self.no_of_words, len(docs))).astype(float)

    def SVD_decomposition(self):
        """
        Perform Singular Value Decomposition (SVD) on the term frequency matrix.
        """
        self.U, sing_vals, VT = svds(self.tfs, k=700, which='LM')
        self.V = VT.T
        self.sing_vals_matrix = np.diag(sing_vals)
        self.sing_vals_matrix_inv = np.diag(1 / sing_vals)
        self.Usig_inv = np.dot(self.U, self.sing_vals_matrix_inv)

    def process_query(self, query):
        """
        Process the query by removing stop words and stemming the remaining words.
        """
        query_vec = [0] * len(self.vocab)
        stop_words = set(stopwords.words('english'))
        words_of_q = word_tokenize(query.lower())
        filtered_words = [word for word in words_of_q if word.isalnum() and word not in stop_words]
        for word in filtered_words:
            stemmed_word = self.stemmer.stem(word)
            if stemmed_word in self.vocab:
                query_vec[self.vocab[stemmed_word]] += 1

        return query_vec

    def calculate_similarities(self, query_vec):
        """
        Calculate the cosine similarity between the query vector and the document vectors.
        """
        lsa_query = np.dot(query_vec, self.Usig_inv)
        norm_query = norm(lsa_query)

        similarities = []
        for i, row in enumerate(self.V):
            similarity = np.dot(row, lsa_query) / (norm(row) * norm_query)
            similarities.append((similarity, i))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities

    def rank(self, docs, docIDs, queries):
        """
        Rank the documents based on their similarity to the queries.
        """
        preprocessed_docs = self.preprocess_docs(docs)
        self.build_vocab(preprocessed_docs)
        self.build_index(preprocessed_docs)
        self.SVD_decomposition()

        docIDs_ordered = []
        for query in queries:
            query_vec = self.process_query(query)
            similarities = self.calculate_similarities(query_vec)
            ordered_docs = [docIDs[i] for _, i in similarities]
            docIDs_ordered.append(ordered_docs)

        return docIDs_ordered