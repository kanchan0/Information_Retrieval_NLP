# from util import *
from itertools import count
import numpy as np
from math import log2
# Add your import statements here
import math



class Evaluation():

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

        count = 0
        for id in query_doc_IDs_ordered[:k]:
            if( int(id) in true_doc_IDs):
                count += 1
        return count/k


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

        num_q = len(query_ids)
        precisions = []

        for x in range(num_q):
            q_doc = doc_IDs_ordered[x]
            q_id = int(query_ids[x])
            original_ids = []

            for val in qrels:
                if(int(val["query_num"]) == int(q_id)):
                    original_ids.append(int(val["id"]))
            prec = self.queryPrecision(q_doc, q_id, original_ids, k)
            precisions.append(prec)
        
        return sum(precisions)/len(precisions)

    
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

        n_original_docs = len(true_doc_IDs)
        count = 0
        for id in query_doc_IDs_ordered[:k]:
            if(int(id) in true_doc_IDs):
                count += 1

        return count/n_original_docs

        
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

        n_queries = len(query_ids)
        recalls = []

        for x in range(n_queries):
            q_doc = doc_IDs_ordered[x]
            q_id = query_ids[x]
            original_ids = []
            for val in qrels:
                if(int(val["query_num"]) == int(q_id)):
                    original_ids.append(int(val["id"]))
            recall = self.queryRecall(q_doc, q_id, original_ids, k)
            recalls.append(recall)

        return sum(recalls)/len(recalls)
        

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

        f_score = 0
        prec = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        if(prec > 0 and recall > 0):
            f_score = (2 * prec * recall)/ (prec + recall)

        return f_score

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

        n_queries = len(query_ids)
        f_scores = []
        for x in range(n_queries):
            q_doc = doc_IDs_ordered[x]
            q_id = query_ids[x]
            original_ids = []

            for val in qrels:
                if(int(val["query_num"]) == int(q_id)):
                    original_ids.append(int(val["id"]))
            f_score = self.queryFscore(q_doc, q_id, original_ids, k)
            f_scores.append(f_score)

        return sum(f_scores)/len(f_scores)

        
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
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        n_docs = len(query_doc_IDs_ordered)
        relevant_docs = []
        relevant_values = {}
        DCG_k = 0
        IDCG_k = 0

        for val in true_doc_IDs:
            if(int(val["query_num"]) == int(query_id)):
                q_id = int(val["id"])
                rel = 5 - val["position"]
                relevant_values[int(q_id)] = rel
                relevant_docs.append(int(q_id))
        
        for x in range(1, k+1):
            d_id = int(query_doc_IDs_ordered[x-1])
            if d_id in relevant_docs:
                rel = relevant_values[d_id]
                DCG_k += ((2**rel) - 1) / log2(x+1)  
        
        ord_values = sorted(relevant_values.values(), reverse=True)
        n_docs = len(ord_values)

        for x in range(1, min(n_docs, k)+1):
            rel = ord_values[x-1]
            IDCG_k += ((2**rel)-1)/log2(x+1)
        
        return DCG_k / IDCG_k

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
    
        n_queries = len(query_ids)
        nDCG = []
        for x in range(n_queries):
            q_doc = doc_IDs_ordered[x]
            q_id = int(query_ids[x])
            nDCG_x = self.queryNDCG(q_doc, q_id, qrels, k)
            nDCG.append(nDCG_x)

        return sum(nDCG)/len(nDCG)


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
        rels = []
        precs = []
        for id in query_doc_IDs_ordered:
            if(int(id) in true_doc_IDs):
                rels.append(1)
            else:
                rels.append(0)
        
        for x in range(1, k+1):
            prec = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, x)
            precs.append(prec)
        
        prec_k = []
        for i in range(k):
            val = precs[i]*rels[i]
            prec_k.append(val)
        
        if(sum(rels[:k]) != 0):
            avg_prec = sum(prec_k)/sum(rels[:k])
        else:
            avg_prec = 0
        
        return avg_prec


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

        avg_precs = []
        n_queries = len(query_ids)
        for i in range(n_queries):
            q_doc = doc_IDs_ordered[i]
            q_id = int(query_ids[i])
            original_ids = []

            for val in q_rels:
                if int(val["query_num"]) == int(q_id):
                    original_ids.append(int(val["id"]))
            avg_prec = self.queryAveragePrecision(
                q_doc, q_id, original_ids, k)
            avg_precs.append(avg_prec)

        return sum(avg_precs)/len(avg_precs)


