# Add your import statements here
from util import *
import numpy as np
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from collections import defaultdict

def cosine_similarity(vector1, vector2):
	'''
		This function calculate cosine similarity between two given inputs
		INPUTS : two vectors, vector1 & vector2
		Output : cosine similarity between the input vectors.
	'''

	ep = 1e-4	# this small number is added to vectors to avoid division with zero	
	vector1 = np.array(vector1) 
	vector2 = np.array(vector2)
	vector1 += np.full(vector1.shape, ep) #added epsilon to each element of vector to avoid exception error due to division with zero
	vector2 += np.full(vector2.shape, ep)
	return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2)) 

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		self.index = None
		#Fill in code here
		#print(docs)
		#print(docIDs)
		iterator = 0
		index_ = {}
		words_in_doc = {}
		self.docIDs = docIDs
		self.stemmer = PorterStemmer()
		tf_table = defaultdict(defaultdict)

		for doc in docs:
			words_in_doc[docIDs[iterator]] = 0
			for sentence in doc:
				for words in sentence:
					word_stemmed = self.stemmer.stem(words)
					if not word_stemmed in index_.keys():
						index_[word_stemmed] = [docIDs[iterator]]
					else:
						if not docIDs[iterator] in index_[word_stemmed]:
							index_[word_stemmed].append(docIDs[iterator])
					
					temp = docIDs[iterator]
					if not word_stemmed in tf_table[temp].keys():
						tf_table[temp][word_stemmed] = 1
					else:
						tf_table[temp][word_stemmed] = tf_table[temp][word_stemmed] + 1
					
					words_in_doc[temp] = words_in_doc[temp] + 1
			iterator+= 1


		
		'''
		tf-idf value is calculated by first computing the termfrequency normalized by the 
		total number of words in the document, multiplied by the IDF,which is calculated as
		the logarithm of the ratio of the total number of documents to the number of documents 
		containing the term.

		Normalizing the term frequency (TF) is important in TF-IDF calculations because it 
		helps to mitigate the effect of document length on the TF component of the TF-IDF score.
		Without normalization, longer documents tend to have higher term frequencies simply
		because they contain more words, which can bias the importance of terms within those 
		documents.
		'''
		iterator = 0
		self.tf_idf = {}
		for doc in tf_table.keys():
			self.tf_idf[doc] = [0.0] * len(index_.keys())
			for word in tf_table[doc].keys():
				ind = list(index_.keys()).index(word)
				self.tf_idf[doc][ind] = (tf_table[doc][word] / (words_in_doc[docIDs[iterator]] + 1)) * np.log(len(docIDs)/ len(index_[word]))
			iterator+= 1

		self.index = index_

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query
		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		#Fill in code here

		query_term_freq = {}
		AllComparisons = []
		all_words = list(self.index.keys())

		for query in queries:
			ctr = 0 
			for sentence in query:
				for term in sentence:
					word_stemmed = self.stemmer.stem(term)
					ctr += 1
					if not word_stemmed in all_words:
						continue
					if not word_stemmed in query_term_freq:
						query_term_freq[word_stemmed] = 1
					else:
						query_term_freq[word_stemmed] += 1
			
			query_vec = [0] * len(self.index.keys())
			for word in query_term_freq.keys():

				indii = list(self.index.keys()).index(word)
				query_vec[indii] = (query_term_freq[word] / ctr) * np.log(len(self.docIDs) / len(self.index[word]))
			
			ctr = 0
			query_term_freq = {}

			'''
			the cosine similarity between the query vector and the TF-IDF vectors of all documents. 
			We then sort the documents based on their similarity to the query in descending order 
			and appending the ordered list of document IDs to all_comparisons.
			'''

			sim_scores = {}
			keys =  self.tf_idf.keys()
			for doc in keys:
				x = self.tf_idf[doc]
				sim_scores[doc] = cosine_similarity(x, query_vec)
		
			sim_scores = dict(sorted(sim_scores.items(), key=lambda item: item[1], reverse=True))
			#print(">>>>>>>>>>",(sim_scores==sim_))
			AllComparisons.append(list(sim_scores.keys()))

		doc_IDs_ordered = AllComparisons
		return doc_IDs_ordered