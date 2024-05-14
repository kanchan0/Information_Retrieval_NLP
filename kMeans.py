import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import numpy as np
from tfidf import TF_IDF

class kmeans:
    def build_word_index(self, docs):
        corpora = [] # a list of words
        #word_map = {}
        for doc in docs:
            for sent in doc:
                for word in sent:
                    corpora.append(word)
        corpora = set(corpora)  # unique words
        word_map = {word : idx for idx,word in enumerate(set(corpora),0)} # for assigning a unique label to each word

        return word_map

    def rank(self, docs, doc_ids, queries):
        word_map_doc = self.build_word_index(docs)
        word_map_q = self.build_word_index(queries)
        # TFIDF representation of the documents
        # from tfidf import TF_IDF
        tf_idf_docs = TF_IDF(docs, doc_ids, word_map_doc, normalize = True)
        tf_idf_query = TF_IDF(queries, doc_ids, word_map_q, normalize = True)


        # tf-idf representation of documents
        df_cluster = tf_idf_docs.T
        # silhoutte scores to find the appropriate
        sil_scores = []
        # tuning number of clusters using grid search, from 2 to 11
        for n_clusters in range(2,11):
            # Applying K-means
            clusterer = KMeans(n_clusters=n_clusters)
            preds = clusterer.fit_predict(df_cluster) #return the class_label for each data_point

            score = silhouette_score(df_cluster, preds)
            sil_scores.append(score)
            print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

        plt.plot(list(range(2,11)),sil_scores, marker='o',color= 'r')
        plt.title("Average Silhoutte distance Vs n_clusters")
        plt.show()

        # calculate distortion for a range of number of cluster
        distortions = []
        # Elbow plot to find the optimal clusters
        for i in range(1, 100, 10):
            km = KMeans(n_clusters=i, init='random', n_init=10, random_state=0)
            km.fit(tf_idf_docs.T)
            distortions.append(km.inertia_)

        # plot
        plt.plot( list(range(1,100,10)), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()


        #-----------------------------------------------------------------------------------

        # Best k = 6
        km = KMeans(n_clusters= 6, random_state=0)
        # fitting KMC on documents
        km.fit(tf_idf_docs.T)
        # km.cluster_centers_.shape

        cluster_doc_ids = {}
        # assigning documents to various clusters
        for i in range(1400): # iterate over all the docs and create a dictonary to map docs to clusters
            try :
                cluster_doc_ids[km.labels_[i]] += [i]
            except :
                cluster_doc_ids[km.labels_[i]] = [i]

        # normal method of retrieval
        tic = time.time()
        # for finding the average retrieval time for a query.
        # iterating over queries
        for j in range(200):
            cosine_sim_clust = []
            # iterating over all documents
            for i in range(1400):
                # cosine similarity
                cosine_sim_clust.append(np.matmul(tf_idf_docs[:,i].T, tf_idf_query[:,j]))
            cosine_sim_clust = np.array(cosine_sim_clust)
            doc_IDs_ordered_clust = (np.argsort(cosine_sim_clust,axis=0)+1)[::-1].T.tolist()
        toc = time.time()
        print("without clustering, Average Retrieval time : "+str((toc-tic)/200))

        # clustering method
        tic = time.time()
        doc_IDs_ordered_kmeans = []
        # iterating over queries
        for j in range(200):
            cluster = np.argmax(np.matmul(tf_idf_query[:,j].T, km.cluster_centers_.T))
            cluster_docs = tf_idf_docs[:, cluster_doc_ids[cluster]]
            cosine_sim = np.matmul(cluster_docs.T,tf_idf_query[:, j])
            doc_IDs_ordered_clust = (np.argsort(cosine_sim,axis=0))[::-1].T.tolist()    # contains docID of highest cosine sim for that query
            doc_IDs_ordered_kmeans.append(doc_IDs_ordered_clust)
            #doc_IDs_ordered = np.array(cluster_doc_ids[cluster])[doc_IDs_ordered_clust]+1
        toc = time.time()
        print("clustering method, Retrieval time : "+str((toc-tic)/200))

        # we store the retrieval time taken for each query and store them as a list
        without_clust = []
        # iterating over 200 queries
        for j in range(200):
            tic = time.time()
            cosine_sim = []
            for i in range(1400):
                cosine_sim.append(np.matmul(tf_idf_docs[:,i].T, tf_idf_query[:,j]))
            cosine_sim = np.array(cosine_sim)
            doc_IDs_ordered = (np.argsort(cosine_sim,axis=0)+1)[::-1].T.tolist()
            toc = time.time()
            without_clust.append((toc-tic))

        # print(len(doc_IDs_ordered))

        #print("without clustering, Average Retrieval time : "+str((toc-tic)/200))

        with_clust = []
        # clustering method
        for j in range(200):
            tic = time.time()
            cluster = np.argmax(np.matmul(tf_idf_query[:,j].T, km.cluster_centers_.T))
            cluster_docs = tf_idf_docs[:, cluster_doc_ids[cluster]]
            cosine_sim = np.matmul(cluster_docs.T,tf_idf_query[:,j])
            doc_IDs_ordered_clus = (np.argsort(cosine_sim,axis=0))[::-1].T.tolist()
            # doc_IDs_ordered = np.array(cluster_doc_ids[cluster])[doc_IDs_ordered_clus]+1
            toc = time.time()
            with_clust.append((toc-tic))

        # plot
        plt.figure(figsize = (10,5))
        plt.title('Clustering with Kmeans')
        plt.plot(range(200), without_clust, label = 'Without Clustering')
        plt.plot(range(200), with_clust, label = 'With Clustering')
        plt.legend()
        plt.show()