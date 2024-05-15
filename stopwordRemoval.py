from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords



class StopwordRemoval():
	def __init__(self):
		nltk.download('stopwords')   			          # this will download the stop words from nltk
		self.stopwords = set(stopwords.words('english'))  # this is making stopwords in english language to be matched

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer
		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		
		"""
		stopwordRemovedText = None
		#Fill in code here
		stopWordRemoved = []   # empty list as above variable we are not asked to change
		'''
		this itereates through each sentecne in the input list of tokenized documents and filters our tokens that
		are not stopwords and in the list it is appended.
		'''
		for sentence in text:
			filtered_text = [token for token in sentence if token.lower() not in self.stopwords and token not in [".",","]]
			stopWordRemoved.append(filtered_text)
	
		stopwordRemovedText = stopWordRemoved
		#print(stopwordRemovedText)   # for testing the output
		return stopwordRemovedText




	