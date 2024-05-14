from evaluation_for_plotting import EvaluationPlot  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class plotter():
    def plot(model1, model2, query_ids, qrels):

        print("insidde")
        evaluator = EvaluationPlot()

        mprecision_1, precision_1 = evaluator.meanPrecision(
            model1, query_ids, qrels, 10)
        mrecall_1, recall_1 = evaluator.meanRecall(
            model1, query_ids, qrels, 10)
        mfscore_1, fscore_1 = evaluator.meanFscore(
            model1, query_ids, qrels, 10)
        mMAP_1, MAP_1 = evaluator.meanAveragePrecision(
            model1, query_ids, qrels, 10)
        mnDCG_1, nDCG_1 = evaluator.meanNDCG(
            model1, query_ids, qrels, 10)

        mprecision_2, precision_2 = evaluator.meanPrecision(
            model2, query_ids, qrels, 10)
        mrecall_2, recall_2 = evaluator.meanRecall(
            model2, query_ids, qrels, 10)
        mfscore_2, fscore_2 = evaluator.meanFscore(
            model2, query_ids, qrels, 10)
        mMAP_2, MAP_2 = evaluator.meanAveragePrecision(
            model2, query_ids, qrels, 10)
        mnDCG_2, nDCG_2 = evaluator.meanNDCG(
            model2, query_ids, qrels, 10)

        # Precision graph
        x_label = 'Precision @ k = ' + str(10)
        plt.figure(figsize=(10, 5))
        plt.xlabel(x_label)
        plt.ylabel('Number of Queries')
        plt.title('Precision: VSM vs W2V')
        sns.histplot(precision_1, bins=10, color='blue', alpha=0.5, label='VSM', kde=True)
        sns.histplot(precision_2, bins=10, color='red', alpha=0.5, label='W2V', kde=True)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig("output/comp_prec_W2V.png")

        # Recall graph
        x_label = 'Recall @ k = ' + str(10)
        plt.figure(figsize=(10, 5))
        plt.xlabel(x_label)
        plt.ylabel('Number of Queries')
        plt.title('Recall: VSM vs W2V')
        sns.histplot(recall_1, bins=10, color='blue', alpha=0.5, label='VSM', kde=True)
        sns.histplot(recall_2, bins=10, color='red', alpha=0.5, label='W2V', kde=True)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig("output/comp_Recall_W2V.png")

        # F-score graph
        x_label = 'F-score @ k = ' + str(10)
        plt.figure(figsize=(10, 5))
        plt.xlabel(x_label)
        plt.ylabel('Number of Queries')
        plt.title('F-score: VSM vs W2V')
        sns.histplot(fscore_1, bins=10, color='blue', alpha=0.5, label='VSM', kde=True)
        sns.histplot(fscore_2, bins=10, color='red', alpha=0.5, label='W2V', kde=True)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig("output/comp_Fscore_W2V.png")

        # MAP graph
        x_label = 'MAP @ k = ' + str(10)
        plt.figure(figsize=(10, 5))
        plt.xlabel(x_label)
        plt.ylabel('Number of Queries')
        plt.title('MAP: VSM vs W2V')
        sns.histplot(MAP_1, bins=10, color='blue', alpha=0.5, label='VSM', kde=True)
        sns.histplot(MAP_2, bins=10, color='red', alpha=0.5, label='W2V', kde=True)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig("output/comp_MAP_W2V.png")

        # nDCG graph
        x_label = 'nDCG @ k = ' + str(10)
        plt.figure(figsize=(10, 5))
        plt.xlabel(x_label)
        plt.ylabel('Number of Queries')
        plt.title('nDCG: VSM vs W2V')
        sns.histplot(nDCG_1, bins=10, color='blue', alpha=0.5, label='VSM', kde=True)
        sns.histplot(nDCG_2, bins=10, color='red', alpha=0.5, label='W2V', kde=True)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig("output/comp_nDCG_W2V.png")
