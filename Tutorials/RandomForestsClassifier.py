import numpy as np
import time

from RandomizedTree import RandomizedTree

class RandomForestsClassifier(object):
	def __init__(self, numTrees=10, stopCriteria='maxDepth', stopValue=2, subspaceSize=500):

		self.numTrees = numTrees
		self.stopCriteria = stopCriteria
		self.stopValue = stopValue
		self.subspaceSize = subspaceSize
		
		self.trees = []

	def train(self, X, y, Xtest, ytest):
		self.numClasses = len(np.unique(y))
		self.inputSize = X.shape[1]
		
		(self.labelMap, y) = self.mapLabels(y.flatten())
	   
		totalScores = np.zeros((X.shape[0], self.numClasses))
		totalTestScores = np.zeros((Xtest.shape[0], self.numClasses))

		for idx in range(self.numTrees):
			print 'Building tree: %d/%d'%(idx+1, self.numTrees)

			start = time.time()
			rt = RandomizedTree(self.subspaceSize,self.stopCriteria, self.stopValue)
			rt.train(X, y)
			print 'trained in ', time.time() - start

			# train accuracy
			start = time.time()
			(pred, scores) = rt.predict(X)
			accuracy = 1 - 1.0*sum(pred != y.flatten())/len(y)
			totalScores = totalScores + scores
			idxes = totalScores.argmax(axis=1)
			totalAccuracy = 1 - 1.0*sum(idxes != y.flatten())/len(y)
			print 'train prediction', time.time() - start

			# test accuracy
			start = time.time()
			(pred, scores) = rt.predict(Xtest)
			pred = self.mapbackLabels(pred)
			testAccuracy = 1 - 1.0*sum(pred != ytest.flatten())/len(ytest)
			totalTestScores = totalTestScores + scores
			idxes = totalTestScores.argmax(axis=1)
			idxes = self.mapbackLabels(idxes)
			totalTestAccuracy = 1 - 1.0*sum(idxes != ytest.flatten())/len(ytest)
			print 'test prediction', time.time() - start

			self.trees.append(rt)
			 
			print 'Training Completed with Train Accuracy: %f, Test Accuracy: %f'%(accuracy, testAccuracy)
			print 'Combined Train Accuracy: %f, Test Accuracy: %f'%(totalAccuracy, totalTestAccuracy)
	
	def predictBest(self, X, N=None):
		if N is None or self.numClasses < N:
			N = self.numClasses

		alldata = np.zeros((self.numTrees, self.numClasses))
		scores = np.zeros((X.shape[0], self.numClasses))

		for idx, rt in enumerate(self.trees):
		   (pred1, scores1) = rt.predict(X, None, False) 
		   alldata[idx, :] = scores1[:]
		   scores = scores + scores1

		scores = scores/self.numTrees
		pred = scores.argmax(axis=1)

		mappedBackPred = self.mapbackLabels(pred)
		pred = np.zeros(scores.shape)

		for idx in range(scores.shape[0]):
			sortIdx = (-scores[idx,:]).argsort()

			enumScores = scores[idx,:][sortIdx]
			pred[idx,:] = np.transpose(sortIdx)
			scores[idx,:] = np.transpose(enumScores)

		pred = pred.astype('int')
		pred = self.mapbackLabels(pred[:, 0:N])
		scores = scores[:, 0:N]

		return (pred, scores, alldata)

	def predictTopN(self,X,N=None):
		if N is None or self.numClasses < N:
			N = self.numClasses

		(mappedBackPred, scores) = self.predict(X)

		pred = np.zeros(scores.shape)

		for idx in range(scores.shape[0]):
			sortIdx = (-scores[idx,:]).argsort()

			enumScores = scores[idx,:][sortIdx]
			pred[idx,:] = np.transpose(sortIdx)
			scores[idx,:] = np.transpose(enumScores)

		pred = self.mapbackLabels(pred[:, 0:N])
		scores = scores[:, 0:N]

		return (pred, scores)

	def predictAll(self,X):
		scores = np.zeros((X.shape[0] ,self.numClasses))

		for rt in self.trees:
		   (pred1, scores1) = rt.predict(X) 
		   scores = scores + scores1

		scores = scores/self.numTrees
		pred = self.labelMap

		return (pred, scores)

	def predict(self, X):
		scores = np.zeros((X.shape[0], self.numClasses))

		for rt in self.trees:
		   (pred1, scores1) = rt.predict(X)
		   scores = scores + scores1
		
		scores = scores/self.numTrees
		pred = scores.argmax(axis=1)
		pred = self.mapbackLabels(pred)

		return (pred, scores)
	
	def mapLabels(self, yIn):
		labelMap = np.unique(yIn).astype('int32')
		mappedY = np.zeros(yIn.shape, dtype='int32').flatten()
		for idx in range(len(labelMap)):
			mappedY[yIn == labelMap[idx]] = idx
		
		return (labelMap, mappedY)
	
	def mapbackLabels(self, y):
		return self.labelMap[y]

	def save(self):
		data = {}
		data['numTrees'] = self.numTrees
		data['labelMap'] = self.labelMap
		data['numClasses'] = self.numClasses
		data['trees'] = []

		for idx, rt in enumerate(self.trees):
			data['trees'].append(rt.save())

		return data

	def load(self, data):
		self.numTrees = data['numTrees']
		self.labelMap = data['labelMap']
		self.numClasses = data['numClasses']

		self.trees = []
		trees = data['trees']
		for tree in trees:
			rt = RandomizedTree()
			rt.load(tree)

			# raw_input()
			self.trees.append(rt)
