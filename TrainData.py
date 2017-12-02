import glob
import numpy as np
from PreProcessData import PreProcessData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler

class TrainData():
	#contains all the training data
	all_training_data=[]
	#contains all the training labels
	all_training_labels=[]
	
	#contains all the training data
	all_test_data=[]
	#contains all the training labels
	all_test_labels=[]

	tfidf_vectorized_data=[]
	tfidf_vectorizer = TfidfVectorizer(analyzer="word")	
	
	ppd = PreProcessData()
	
	sgdc_tfidf = SGDClassifier(loss="hinge", penalty="l2")
	mnb_tfidf = MultinomialNB()
	gnb_tfidf = GaussianNB()#
	forest_tfidf = RandomForestClassifier(n_estimators=100)
	
	all_models={}
	all_test_data_models={}
	
	def trainOnDataSet(self):
		for file in glob.glob("bbc/*/**.txt"):
		
			file_category = file.split('\\')[1]
			processes_file_data = self.ppd.processFile(file)
			
			self.all_training_data.append(processes_file_data)
			self.all_training_labels.append(self.categoryToNumeric(file_category))
		print(len(self.all_training_data))
		
	def createVectors(self):
		self.tfidf_vectorized_data = self.tfidf_vectorizer.fit_transform(self.all_training_data)
		self.tfidf_vectorized_data = self.tfidf_vectorized_data.toarray()
			
	def prepareModels(self):
		
		#rbf_feature = RBFSampler(gamma=1, random_state=1)
		#vectorized_data = rbf_feature.fit_transform(self.all_training_data)	
		#vectorized_data = self.vectorizer.fit_transform(self.all_training_data)
		#vectorized_data = vectorized_data.toarray()
		# if you want to see the vocabulary
		#vocab = vectorizer.get_feature_names()
		#top_10 = 
		#print(vocab)

		self.forest_tfidf = self.forest_tfidf.fit(self.tfidf_vectorized_data, self.all_training_labels)
		self.sgdc_tfidf = self.sgdc_tfidf.fit(self.tfidf_vectorized_data, self.all_training_labels)
		self.mnb_tfidf = self.mnb_tfidf.fit(self.tfidf_vectorized_data, self.all_training_labels)
		self.gnb_tfidf = self.gnb_tfidf.fit(self.tfidf_vectorized_data, self.all_training_labels)
		
		self.all_models['random_forest_tfidf']= self.forest_tfidf
		self.all_models['sgdc_tfidf'] = self.sgdc_tfidf
		self.all_models['mnb_tfidf'] = self.mnb_tfidf
		self.all_models['gnb_tfidf'] = self.gnb_tfidf		
		
		return self.all_models

		#print ('this is: '+str(self.forest))
		#return self.forest
		
	def getTopFeatures(self):
		#tfidf_feature_names = self.tfidf_vectorized_data.toarray()
		tfidf_feature_names = self.tfidf_vectorized_data.get_feature_names()
		
		fea = open('top_features.txt','w')
		for k,v in all_models.items():
			fea.write(k)
			for i, class_label in enumerate(self.all_training_labels):
				top10 = np.argsort(v.coef_[i])[-10:]
				fea.write("%s: %s" % (class_label," ".join(tfidf_feature_names[j] for j in top10)))

	def prediction(self, test_all_data, predictor, s, pnr):
		print(len(self.all_test_data))
		correct=0
		total=0
		real=0
		fake=0
		real_correct=0
		fake_correct=0
		for n in range(0,len(test_all_data)):
			prediction = predictor.predict(test_all_data[n])
			prediction = prediction[0]
			if prediction == 1:
				real +=1
			if prediction == 2:
				fake +=1
			if prediction == self.all_test_labels[n]:
				correct +=1
				if prediction == 1:
					real_correct +=1
				if prediction == 2:
					fake_correct +=1
			else:
				print('wrong prediction '+str(prediction)+' '+ str(self.all_test_labels[n]))
				#pnr.write('wrong prediction - prediction: '+str(self.labelToCategory(prediction))+' correct: '+ str(self.labelToCategory(self.all_test_labels[n])))
				#pnr.write('\n')
			total +=1
		pnr.write('total real predictions:'+str(real))
		pnr.write('\n')
		pnr.write('total fake predictions:'+str(fake))
		pnr.write('\n')			
		pnr.write('total correct real predictions:'+str(real_correct))
		pnr.write('\n')
		pnr.write('total correct fake predictions:'+str(fake_correct))
		pnr.write('\n')
		#print ('prediction: ', self.labelToCategory(prediction))
		result = str(s)+'\n Accuracy is '+str(correct/total)
		#res.close()
		return result
		
	def prepareTestData(self):
		for file in glob.glob("test/*/**.txt"):
		
			test_file_category = file.split('\\')[1]
			processed_file_data = self.ppd.processFile(file)
			
			self.all_test_data.append(processed_file_data)
			self.all_test_labels.append(self.categoryToNumeric(test_file_category))
			#all_test_labels.append(categoryToNumeric(file_category))
		#return self.vectorize()
			
	def vectorizeTestData(self):		
		vectorized_data_tfidf = self.tfidf_vectorizer.transform(self.all_test_data)
		vectorized_data_tfidf = vectorized_data_tfidf.toarray()
		
		#self.all_test_data_models['count vectorized'] = vectorized_data_count
		self.all_test_data_models['tfidf vectorized'] = vectorized_data_tfidf
		
		return self.all_test_data_models
		
		# if you want to see the vocabulary
		#vocab = vectorizer.get_feature_names()
		#print(vocab)
		#forest = forest.fit(vectorized_data, all_test_labels)
		#return (vectorized_data)
		
	def categoryToNumeric(self,category):
		if category == 'business' or category == 'entertainment' or category == 'politics' or category == 'sport' or category == 'tech':
			return 1
		if category == 'fake':
			return 2			

	def labelToCategory(self,label):
		if label == 1:
			return 'real'
		if label == 2:
			return 'fake'

	def main(self):
		print('training on dataset')
		self.trainOnDataSet()
		print('training on dataset----complete')
		print('creating vectors')		
		self.createVectors()
		print('creating vectors----complete')
		print('creating models')
		am = self.prepareModels()
		print('creating models----complete')
		print('preparing test data')		
		self.prepareTestData()
		print('preparing test data----complete')		
		print('vectorizing test data')	
		atdm = self.vectorizeTestData()
		print('vectorizing test data----complete')	
		res = open('result.txt','w')
		pnr = open('pandr.txt','w')
		for k,v in am.items():
			for name, model in atdm.items():
				s = 'Predicting for Model: ' +str(k) + ' Test Model is: '+str(name)
				pnr.write(s)
				pnr.write('\n')
				result = self.prediction(model, v, s, pnr)
				res.write(str(result))
				res.write('\n')
		res.close()
		pnr.close()
		#self.getTopFeatures()