from nltk.corpus import stopwords
from itertools import chain
import math

class BestMatch_25():

    def __init__(self):
        self.index = None
        self.docIDs = None

    def rank(self, docs, doc_ids, queries):
        # Concatenating nested lists within docs to create a list of texts
        texts = [list(chain.from_iterable(docs[i])) for i in range(len(docs))]

        class BM_25:
            def __init__(self, k=1.85, b=0.8):
                # Initializing BM25 parameters
                self.b = b
                self.k = k
            
            def fit(self, corpus):
                # Initializing lists to store term frequencies, document frequencies, and other statistics
                term_freq = []
                doc_freq = {}
                inv_doc_freq = {}
                list_doc_length = []
                corpus_size = 0

                # Calculating document statistics
                for doc in corpus:
                    corpus_size += 1
                    list_doc_length.append(len(doc))
                    frequencies = {}
                    # Counting term frequencies within each document
                    for token in doc:
                        term_count = frequencies.get(token, 0) + 1
                        frequencies[token] = term_count

                    term_freq.append(frequencies)
                    # Updating document frequency counts
                    for token, _ in frequencies.items():
                        df_count = doc_freq.get(token, 0) + 1
                        doc_freq[token] = df_count

                # Computing inverse document frequencies
                for token, freq in doc_freq.items():
                    inv_doc_freq[token] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

                # Storing computed statistics
                self.tf = term_freq
                self.idf = inv_doc_freq
                self.document_length_list = list_doc_length
                self.size_corpus = corpus_size
                self.avg_length_list = sum(list_doc_length) / corpus_size
                return self

            # Computing scores for the given query
            def computation(self, query):
                scores = [self.score_calculator(query, i) for i in range(self.size_corpus)]
                return scores

            # Calculating BM25 score for a specific document
            def score_calculator(self, query, index):
                score = 0.0
                list_doc_length = self.document_length_list[index]
                frequencies = self.tf[index]
                for token in query:
                    if token not in frequencies:
                        continue
                    freq = frequencies[token]
                    num = self.idf[token] * freq * (self.k + 1)
                    den = freq + self.k * (1 - self.b + self.b * list_doc_length / self.avg_length_list)
                    score += (num / den) 
                return score

        doc_IDs_odered_BM_25 = []
        
        # Iterating over each query
        for query in queries:
            # Preprocessing query text by converting to lowercase and removing stopwords
            query = [word for word in query.lower().split() if word not in stopwords.words('english')]

            # Initializing BM25 object and fitting it with corpus
            bm_25_obj = BM_25()
            bm_25_obj.fit(texts)
            # Computing BM25 scores for the query
            scores = bm_25_obj.computation(query)

            scores_list = []
            # Combining scores with document IDs
            for score, doc in zip(scores, doc_ids):
                final_val_scores = []
                final_val_scores.append(score)
                final_val_scores.append(doc)
                scores_list.append(final_val_scores)
            
            # Ranking documents based on scores
            ranked_docs = sorted(scores_list, key=lambda x: x[0], reverse=True)
            # Extracting only document IDs from ranked documents
            doc_ordering = [ranked_docs[i][1] for i in range(1400)]
            doc_IDs_odered_BM_25.append(doc_ordering)
                
        return doc_IDs_odered_BM_25
