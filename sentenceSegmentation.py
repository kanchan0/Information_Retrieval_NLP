from util import *

# Add your import statements here
import re
import nltk
from nltk.tokenize import sent_tokenize

# downloading punkt tokenizer if not already downloaded for current environment
nltk.download('punkt')

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		#Fill in code here
		delimeters = r'(?<=[.!?:])+'
		sentences = re.split(delimeters, text)

		#we are removing empty strings from the list
		sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
		segmentedText = sentences
		
		#print(segmentedText) # testing for correct output
		return segmentedText



	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
		
		segmentedText = None

		#Fill in code here
		sentences = sent_tokenize(text)
		segmentedText = sentences
		#print(segmentedText)
	
		return segmentedText