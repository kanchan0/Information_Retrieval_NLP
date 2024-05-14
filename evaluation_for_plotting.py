import math
import pandas as pd
import numpy as np
class EvaluationPlot():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		# Count the number of calculated doc ids that matches true doc ids for that query 
		precision = sum(1 for doc_id in query_doc_IDs_ordered[:k] if int(doc_id) in true_doc_IDs)

		return precision/ k


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = 0

		no_of_queries = len(query_ids)

		precision = []

		for i in range(no_of_queries):

			relevant_document = doc_IDs_ordered[i]
			query_id = int(query_ids[i])

			#Create a list of true document ids information for the particular query available in cran_qrels.json
			true_ids = [int(val["id"]) for val in qrels if int(val["query_num"]) == query_id]

			precise = self.queryPrecision(relevant_document, query_id, true_ids, k)

			precision.append(precise)
        
		meanPrecision = sum(precision)/len(precision)
		
		return meanPrecision, precision


	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""
		
		recall = -1
		
		# Count the number of calculated doc ids that matches true doc ids for that query 
		recall = sum(1 for doc_id in query_doc_IDs_ordered[:k] if int(doc_id) in true_doc_IDs)

		#For precision calculation we divide by no of retrieved documents. Note the difference here

		if len(true_doc_IDs) == 0:
			recall = 0
		else:
			recall = recall/ len(true_doc_IDs)
		return recall
	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		no_of_queries = len(query_ids)

		recall = []

		for i in range(no_of_queries):

			relevant_document = doc_IDs_ordered[i]
			query_id = int(query_ids[i])

			#Create a list of true document ids information for the particular query available in cran_qrels.json
			true_ids = [int(val["id"]) for val in qrels if int(val["query_num"]) == query_id]

			rec = self.queryRecall(relevant_document, query_id, true_ids, k)

			recall.append(rec)
        
        # Compute the mean recall
		meanRecall = sum(recall)/len(recall)

		return meanRecall, recall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		
		#To avoid division by zero error
		if (precision + recall) == 0:
			fscore = 0
		else:
			fscore = 2*precision*recall/(precision + recall)
                        
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0

		no_of_queries = len(query_ids)

		Fscore = []

		for i in range(no_of_queries):

			relevant_document = doc_IDs_ordered[i]
			query_id = int(query_ids[i])

			#Create a list of true document ids information for the particular query available in cran_qrels.json
			true_ids = [int(val["id"]) for val in qrels if int(val["query_num"]) == query_id]

			f = self.queryFscore(relevant_document, query_id, true_ids, k)

			Fscore.append(f)
        
		meanFscore = sum(Fscore)/len(Fscore)

		return meanFscore, Fscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth) # some things wrong here, it should be having relevance rating also, so it must be a dict or df
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = 0

		#Fill in code here
		relevance = np.zeros((len(query_doc_IDs_ordered), 1))
		true_doc_IDs["position"] = 5 - true_doc_IDs["position"] 


		# finding ideal DCG value
		true_doc_IDs_sorted = true_doc_IDs.sort_values("position", ascending = False)
		DCG_ideal = true_doc_IDs_sorted.iloc[0]["position"]

		for i in range(1, min(k,len(true_doc_IDs))):
			DCG_ideal += true_doc_IDs_sorted.iloc[i]["position"] * np.log(2)/np.log(i+1)

		t_doc_IDs = list(map(int, true_doc_IDs["id"]))
		for i in range(k):
			if query_doc_IDs_ordered[i] in t_doc_IDs:
				relevance[i] = true_doc_IDs[true_doc_IDs["id"] == str(query_doc_IDs_ordered[i])].iloc[0]["position"]

		for i in range(k):
			nDCG += relevance[i] * np.log(2) / np.log(i + 2)  # Note that here index starts from 0

		nDCG = nDCG/DCG_ideal

		# print(nDCG)

		return nDCG[0]


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		nDCGs = []

		#Fill in code here
		qrels_df = pd.DataFrame(qrels)
		for i in range(len(query_ids)):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_doc_IDs = qrels_df[["position","id"]][qrels_df["query_num"] == str(query_id)]			
			nDCG = self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

			nDCGs.append(nDCG)
		meanNDCG = sum(nDCGs)/len(query_ids)


		return meanNDCG, nDCGs




	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		relevance = [1 if int(id) in true_doc_IDs else 0 for id in query_doc_IDs_ordered]

		#Calculate precision at each k value
		precision = [self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i) for i in range(1, k + 1)]

		# Filter out precision values only at places where relevance values are 1
		rel_precision = [precision[i] * relevance[i] for i in range(k)]

		avgPrecision = sum(rel_precision) / sum(relevance[:k]) if sum(relevance[:k]) != 0 else 0

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		no_of_queries = len(query_ids)

		averagePrecision = []

		for i in range(no_of_queries):

			relevant_document = doc_IDs_ordered[i]
			query_id = int(query_ids[i])

			#Create a list of true document ids information for the particular query available in cran_qrels.json
			true_ids = [int(val["id"]) for val in q_rels if int(val["query_num"]) == query_id]

			avgp = self.queryAveragePrecision(relevant_document, query_id, true_ids, k)

			averagePrecision.append(avgp)
        
		meanAveragePrecision = sum(averagePrecision)/len(averagePrecision)


		return meanAveragePrecision, averagePrecision

