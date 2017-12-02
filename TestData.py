import glob
from PreProcessData import PreProcessData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

class TestData():
	#contains all the training data
	all_test_data=[]
	#contains all the training labels
	#all_test_labels=[]
	
	ppd = PreProcessData()
	
	def prepareTestData(self):
		for file in glob.glob("test/*/**.txt"):
		
			#file_category = file.split('\\')[1]
			processes_file_data = self.ppd.processFile(file)
			
			self.all_test_data.append(processes_file_data)
			#all_test_labels.append(categoryToNumeric(file_category))
		return self.vectorize()
			
	def vectorize(self):
		vectorizer = CountVectorizer(analyzer="word")
		vectorized_data = vectorizer.transform(self.all_test_data)
		vectorized_data = vectorized_data.toarray()
		# if you want to see the vocabulary
		#vocab = vectorizer.get_feature_names()
		#print(vocab)
		#forest = forest.fit(vectorized_data, all_test_labels)
		return (vectorized_data)
		
	def categoryToNumeric(category):
		if category == 'business':
			return 1
		elif category == 'entertainment':
			return 2
		elif category == 'politics':
			return 3
		elif category == 'sport':
			return 4
		elif category == 'tech':
			return 5