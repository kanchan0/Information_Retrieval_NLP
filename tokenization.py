from util import *

# Add your import statements here
import nltk
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		tokenized =[]
		for sentence in text:
			#splitting on the basis of whitespace,tabs or new lines
			tokens = sentence.split() 
			tokenized.append(tokens)
		tokenizedText = tokenized
		#print(tokenizedText) # test run 
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		tokenized = []
		tokenizer = TreebankWordTokenizer()

		for sentence in text:
			tokens = tokenizer.tokenize(sentence)
			tokenized.append(tokens)

		tokenizedText = tokenized

		#print(tokenizedText)
		return tokenizedText