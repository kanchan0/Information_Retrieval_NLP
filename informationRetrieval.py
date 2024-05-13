# Add your import statements here
from util import *
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

def cosine_similarity(vector1, vector2):
	'''
		This function calculate cosine similarity between two given inputs
		INPUTS : two vectors, vector1 & vector2
		Output : cosine similarity between the input vectors.
	'''

	epsilon = 0.0001	# this small number is added to vectors to avoid division with zero	
	vector1 = np.array(vector1) 
	vector1 = vector1 + np.full(vector1.shape, epsilon) #added epsilon to each element of vector to avoid exception error due to division with zero
	vector2 = np.array(vector2)
	vector2 = vector2 + np.full(vector2.shape, epsilon)
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
		index = None
		#Fill in code here
		#print(docs)
		term_freq_table = defaultdict(defaultdict)
		i = 0
		self.docIDs = docIDs
		index_ = {}
		words_in_doc = {}

		for doc in docs:
			words_in_doc[docIDs[i]] = 0
			for sent in doc:
				for words in sent:
					if words not in index_.keys():
						index_[words] = [docIDs[i]]
					else:
						if(docIDs[i] not in index_[words]):
							index_[words].append(docIDs[i])

					if(words not in term_freq_table[docIDs[i]].keys()):
						term_freq_table[docIDs[i]][words] = 1
					else:
						term_freq_table[docIDs[i]][words] += 1
					
					words_in_doc[docIDs[i]] += 1
			i += 1

		
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
		self.tf_idf = {}
		i = 0
		for doc in term_freq_table.keys():
			self.tf_idf[doc] = [0.0] * len(index_.keys())
			for word in term_freq_table[doc].keys():
				ind = list(index_.keys()).index(word)
				self.tf_idf[doc][ind] = (term_freq_table[doc][word] / (words_in_doc[docIDs[i]] + 1)) * np.log(len(docIDs)/ len(index_[word]))
			i += 1

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

		all_words = list(self.index.keys())
		query_term_freq = {}
		all_comparisions = []

		for query in queries:
			count = 0 
			for sent in query:
				for term in sent:
					count += 1
					if term not in all_words:
						continue
					if(term not in query_term_freq):
						query_term_freq[term] = 1
					else:
						query_term_freq[term] += 1
			
			'''
			below I construct the query vector by calculating the TF-IDF weight for each term in
			the query. The TF-IDF weight is computed as (term frequency / total terms in query) *
			log(total documents / documents containing term). This creates a vector representing the
			query in the same vector space as the documents.
			'''

			query_vec = [0] * len(self.index.keys())
			for word in query_term_freq.keys():
				ind = list(self.index.keys()).index(word)
				query_vec[ind] = (query_term_freq[word] / count) * np.log(len(self.docIDs) / len(self.index[word]))
			
			count = 0
			query_term_freq = {}

			'''
			the cosine similarity between the query vector and the TF-IDF vectors of all documents. 
			We then sort the documents based on their similarity to the query in descending order 
			and appending the ordered list of document IDs to all_comparisons.
			'''

			similarities = {}
			for doc in self.tf_idf.keys():
				similarities[doc] = cosine_similarity(self.tf_idf[doc], query_vec)
			
			similarities = {k : v for k, v in sorted(similarities.items(), key= lambda item : item[1], reverse=True)}
			all_comparisions.append(list(similarities.keys()))

		doc_IDs_ordered = all_comparisions
		#print(doc_IDs_ordered)
		return doc_IDs_ordered





