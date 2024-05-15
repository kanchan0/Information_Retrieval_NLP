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
		stopwordRemovedText=[]
		stop_words=set(stopwords.words('english'))

		for sentence in text:
			stopwordRemovedSentence=[]
			for word in sentence:
				if word not in stop_words:
						stopwordRemovedSentence.append(word) #word not in stop words then append in as sentence
			stopwordRemovedText.append(stopwordRemovedSentence)  # append sentence-> text

		return stopwordRemovedText




	