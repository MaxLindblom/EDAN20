import regex as re
import sys
import math
from prettytable import PrettyTable

unigram_frequency = {}
bigram_frequency = {}
total = 0
unigram_probabilities = {}
bigram_probabilities = {}

def tokenize(text):
	text = re.sub(r'\n', "", text)
	sentences = re.findall(r'[A-ZÅÄÖ].+?\.', text)
	sentences = list(map(lambda s: s.lower(), sentences))
	sentences = list(map(lambda s: re.sub(r'[\-,;:!?.’\'«»()–...&‘’“”*—]', "", s), sentences))
	sentences = list(map(lambda s: '<s> ' + s + ' </s>', sentences))
	text = " ".join(sentences)
	return text

def train(text):
	text = tokenize(text)

	global total

	# Unigram calculations
	words = text.split(' ')
	total = len(words)
	for word in words:
		if word in unigram_frequency:
			unigram_frequency[word] += 1
		else:
			unigram_frequency[word] = 1
	for word in unigram_frequency:
		unigram_probabilities[word] = unigram_frequency[word]/total

	# Bigram calculations
	bigrams = [tuple(words[inx:inx + 2])
				for inx in range(len(words) - 1)]
	for bigram in bigrams:
		if bigram in bigram_frequency:
			bigram_frequency[bigram] += 1
		else:
			bigram_frequency[bigram] = 1
	for bigram in bigram_frequency:
		bigram_probabilities[bigram] = bigram_frequency[bigram]/unigram_frequency[bigram[0]]

def unigram_sentence_probability(sentence):
	sentence = re.sub(r'\n', "", sentence)
	sentence = sentence.lower()
	sentence = re.sub(r'[\-,;:!?.’\'«»()–...&‘’“”*—]', "", sentence)
	sentence = sentence + ' </s>'
	words = sentence.split(' ')

	print("----------Unigram probability----------")

	probability = 1
	entropy = 0
	table = PrettyTable(["wi", "C(wi)", "#words", "P(wi)"])
	for word in words:
		table.add_row([
			word, 
			str(unigram_frequency[word]),
			str(total),
			str(unigram_probabilities[word])
		])
		probability = probability*unigram_probabilities[word]
		entropy += math.log2(unigram_probabilities[word])

	print(table)
	
	print("=========================================")
	print("Sentence probability: " + str(probability))

	entropy = -1/len(words)*entropy
	print("Entropy rate: " + str(entropy))

	perplexity = math.pow(2,entropy)
	print("Perplexity: " + str(perplexity))
	print("")

def bigram_sentence_probability(sentence):
	sentence = re.sub(r'\n', "", sentence)
	sentence = sentence.lower()
	sentence = re.sub(r'[\-,;:!?.’\'«»()–...&‘’“”*—]', "", sentence)
	sentence = '<s> ' + sentence + ' </s>'
	words = sentence.split(' ')
	bigrams = [tuple(words[inx:inx + 2])
				for inx in range(len(words) - 1)]


	print("----------Bigram probability----------")

	probability = 1
	entropy = 0
	table = PrettyTable(["wi wi+1", "Ci,i+1", "C(i)", "P(wi+1|wi)", "backoff"])
	for bigram in bigrams:
		bigram_str = str(bigram[0]) + " " + str(bigram[1])
		prob = 0
		freq = 0
		backoff = ""
		if bigram in bigram_probabilities:
			freq = bigram_frequency[bigram]
			prob = bigram_probabilities[bigram]
		else:
			backoff = bigram[1]
			freq = 0
			prob = unigram_probabilities[bigram[1]]

		table.add_row([
			bigram_str, 
			str(freq),
			str(unigram_frequency[bigram[0]]),
			str(prob),
			backoff
		])

		probability = probability*prob
		entropy += math.log2(prob)

	print(table)
	
	print("=========================================")
	print("Sentence probability: " + str(probability))

	entropy = -1/len(bigrams)*entropy
	print("Entropy rate: " + str(entropy))

	perplexity = math.pow(2,entropy)
	print("Perplexity: " + str(perplexity))

def sentence_probability(sentence, probabilities):
	words = re.split(r'\s',sentence)
	print(words)
	probability = 1
	for word in words:
		probability = probability*probabilities[word]
	return probability

if __name__ == '__main__':
	text = sys.stdin.read()
	train(text)

	sentence = "Det var en gång en katt som hette Nils"
	unigram_sentence_probability(sentence)
	bigram_sentence_probability(sentence)