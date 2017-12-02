import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk import ngrams

class PreProcessData():
	
	#process file
	def processFile(self,file):
		
		#read the data in file
		#print(file)
		file_data = open(file,'r',encoding='utf-8', errors='ignore')
		raw_text = file_data.read()
		
		#remove all characters that are not alphabets
		only_alphabets = re.sub("[^a-zA-Z]", " ", raw_text)
		only_alphabets_to_list = only_alphabets.split()	
		#only_alphabets_to_list = ngrams(only_alphabets.split(), 2)
		l=[]
		for x in range(0, len(only_alphabets_to_list)-1):
			l.append(only_alphabets_to_list[x] +" "+only_alphabets_to_list[x+1])
		#stem the words	
		stemmer = SnowballStemmer('english')
		stemmed_words = []
		for word in l:
			stemmed_word = stemmer.stem(word)
			stemmed_words.append(stemmed_word)
		
		#get all stop words from nltk
		stop_words = set(stopwords.words('english'))
		
		#remove stop words from the stemmed words
		only_meaningful_words = [word for word in stemmed_words if word not in stop_words]
			
		#convert list into string
		all_meaningful_words = " ".join(only_meaningful_words)
		
		#close the file
		file_data.close()
		
		return all_meaningful_words