import numpy as np

class RandomizedTree(object):
	# cdef: 
	# 	public int subspaceSize, stopValue, numClasses, inputSize, depth
	# 	public str stopCriteria
	# 	public np.ndarray labelMap, dataIDX, labels, pDist, fSubSpace, splitModel, splitThresh
	# 	public bint terminal
	# 	public RandomizedTree treeRoot, leftNode, rightNode
	# 	public double score

	def __init__(self, subspaceSize=500, stopCriteria='maxDepth', stopValue=2):
		self.subspaceSize = subspaceSize
		self.stopCriteria = stopCriteria
		self.stopValue = stopValue

	def train(self, X, y):
		self.numClasses = len(np.unique(y))
		self.inputSize = X.shape[1]
		self.labelMap, y = self.mapLabels(y)

		dataIDX = np.arange(X.shape[0], dtype=np.int32)
		
		self.treeRoot = self.makeLeafNode(X, y, dataIDX, 1)
		self.treeRoot = self.growTree(self.treeRoot, X, y)
		self.clearTrainData()

	def mapLabels(self, yIn):
		labelMap = np.unique(yIn).astype('int32')
		mappedY = np.zeros(yIn.shape, dtype='int32').flatten()
		for idx in range(len(labelMap)):
			mappedY[yIn == labelMap[idx]] = idx

		return (labelMap, mappedY)

	def mapbackLabels(self, y):
		return self.labelMap[y]

	def makeLeafNode(self, X, y, dataIDX, depth):
		node = RandomizedTree()
		node.dataIDX = dataIDX
		node.depth = depth
		node.terminal = True
		node.pDist = self.computePDist(y[dataIDX])
		node.score = self.computeNodeScore(y[dataIDX])

		node.fSubSpace = None
		node.splitModel = None
		node.splitThresh = None

		node.leftNode = None
		node.rightNode = None

		return node

	def computePDist(self, labels):
		pDist = self.histogram(labels, bins=self.numClasses)
		return 1.0*pDist/pDist.sum()

	def histogram(self,  a,bins=10):
		h = np.zeros(bins, np.int32 )
		N = a.size 

		for i in xrange(N):
			idx = a[i]
			h[idx] += 1

		return h

	def computeNodeScoreOld(self, pDist):
		nzpDist = pDist[pDist > 0]
		return -np.sum(nzpDist * np.log(nzpDist))

	def computeNodeScore(self, labels):
		pDist = np.zeros(self.numClasses, np.double)
		f = 1.0/labels.size 
		N = labels.size

		for i in xrange(N):
			idx = labels[i]
			pDist[idx] += f

		nzPDist = pDist[pDist>0]
		score = -np.sum(nzPDist * np.log(nzPDist))

		return score

	def growTree(self, currNode, X, y):
		#traverses to the leafnodes. Grows the tree at the leaf nodes
		#until the stopCriteria is met
		if currNode.terminal == True:
			grownTree = self.growNode(currNode, X, y)
		else:
			currNode.leftNode = self.growTree(currNode.leftNode, X, y)
			currNode.rightNode = self.growTree(currNode.rightNode, X, y)
			grownTree = currNode

		return grownTree

	def growNode(self, currNode,X, y):

		if not self.shouldStopSplitting(currNode, y):
			(leftDataIDX, rightDataIDX, fSubSpace, splitModel, splitThresh) = self.getBestSplit(currNode, X, y)

			if (leftDataIDX is None) or (rightDataIDX is None) or len(leftDataIDX) == 0 or len(rightDataIDX) == 0:
				return currNode
			
			currNode.leftNode = self.makeLeafNode(X, y, leftDataIDX, currNode.depth+1)
			currNode.leftNode = self.growNode(currNode.leftNode, X, y)

			currNode.rightNode = self.makeLeafNode(X, y, rightDataIDX, currNode.depth+1)
			currNode.rightNode = self.growNode(currNode.rightNode, X, y)

			currNode = self.toInternalNode(currNode, fSubSpace, splitModel, splitThresh)
		else:
			pass

		return currNode

	def shouldStopSplitting(self, currNode, y):
		if self.stopCriteria == 'maxDepth':
			return (currNode.depth >= self.stopValue)
		elif self.stopCriteria == 'minRunnerCount':
			labels = y[currNode.dataIDX]
			(sortedHist, _) = np.histogram(labels, self.numClasses)
			sortedHist.sort()
			sortedHist = sortedHist[::-1]
			return (sortedHist[1] < self.stopValue)
		else:
			assert(False)

	#def [leftDataIDX rightDataIDX fSubSpace splitModel splitThresh] = getBestSplit(self,node):
	def getBestSplit(self, node, X,y ):
		paramSampleSize = self.subspaceSize
		subSpaceDim = 2

		startIdx = 0
		stopIdx = self.inputSize  

		fSets = np.random.choice(np.arange(startIdx,stopIdx,dtype=np.int32), paramSampleSize*subSpaceDim ).reshape(paramSampleSize, subSpaceDim )
		splitModelParams = np.random.normal(0, 1, (paramSampleSize, subSpaceDim+1))
		
		tParams = np.zeros((paramSampleSize, 2))
		tParams[:,1] = -np.Inf
		tParams[:,0] = 2*np.sqrt(2)*np.random.normal(0, 1, paramSampleSize)


		bestGain = 0
		splitGain = 0
		leftDataIDX = None
		leftDataIDX_tmp = None
		rightDataIDX = None
		rightDataIDX_tmp = None
	
		fSubSpace = None
		fSubSpace_tmp = None

		splitModel = None
		splitModel_tmp = None

		splitThresh = None
		splitThresh_tmp = None

		for idx in range(0, paramSampleSize):
			fSubSpace_tmp = fSets[idx, :]
			splitModel_tmp = splitModelParams[idx,:]
			splitThresh_tmp = tParams[idx,:]

			(splitGain, leftDataIDX_tmp, rightDataIDX_tmp) = self.getNodeSplitScore(node, fSubSpace_tmp, splitModel_tmp, splitThresh_tmp,X,y)

			if splitGain > bestGain:
				bestGain = splitGain
				fSubSpace = fSubSpace_tmp

				splitModel = splitModel_tmp 
				splitThresh = splitThresh_tmp

				leftDataIDX = leftDataIDX_tmp
				rightDataIDX = rightDataIDX_tmp

		return (leftDataIDX, rightDataIDX, fSubSpace, splitModel, splitThresh)

	def getNodeSplitScore(self, node, fSubSpace, splitModel, thresh, X, y):

		# If dimension are too big, use index tricks, other a lot of the time will be spent in system time for
		# memory allocation and deallocation
		subSpaceX = X[np.ix_(node.dataIDX, fSubSpace)]
		#cdef np.ndarray[double,ndim=2] subSpaceX = X[node.dataIDX,:][:,fSubSpace]

		fVals = np.dot(subSpaceX, splitModel[:-1]) + splitModel[-1]

		#partition = (fVals < thresh[0]) & (fVals > thresh[1])

		leftDataIDX = node.dataIDX[fVals < thresh[0]]
		rightDataIDX = node.dataIDX[fVals >= thresh[0]]
		#print 'sum matlab %d %d'%(len(leftDataIDX), len(rightDataIDX))
		splitGain = 0.0
		H = 0.0
		HLeft = 0.0
		HRight = 0.0
		#LPDist, RPDist

		if (leftDataIDX.size > 0) and (rightDataIDX.size > 0):
			H = node.score
			# LPDist = self.computePDist(y[leftDataIDX])
			# RPDist = self.computePDist(y[rightDataIDX])

			# HLeft = self.computeNodeScoreOld(LPDist)
			# HRight = self.computeNodeScoreOld(RPDist)

			HLeft = self.computeNodeScore(y[leftDataIDX])
			HRight = self.computeNodeScore(y[rightDataIDX])

			# fprintf('h matlab: #f #f\n', HLeft, HRight)

			splitGain = H - (len(leftDataIDX)*HLeft + len(rightDataIDX)*HRight)/len(node.dataIDX)

			if splitGain < 1e-3:
				splitGain = 0.0 
		
		return (splitGain, leftDataIDX, rightDataIDX)

	def toInternalNode(self, node, fSubSpace, splitModel, splitThresh):
		node.terminal = False
		node.fSubSpace = fSubSpace
		node.splitModel = splitModel
		node.splitThresh = splitThresh
		node.dataIDX = None
		node.pDist = None
		node.score = 0.0

		return node

	def clearTrainData(self):
		self.treeRoot = self.clearLeafData(self.treeRoot)

	def clearLeafData(self, node):
		if not node.terminal:
			node.leftNode = self.clearLeafData(node.leftNode)
			node.rightNode = self.clearLeafData(node.rightNode)
		else:
			node.dataIDX = None

		return node

	def predictTopN(self, X, N = 3):
		(mappedBackPred, scores) = self.predict(X)

		pred = np.zeros(scores.shape)

		for idx in range(0, scores.shape[0]):
			sortIdx = scores[idx,:].argsort()

			enumScores = scores[idx,:][sortIdx]
			pred[idx,:] = np.transpose(sortIdx)
			scores[idx,:] = np.transpose(enumScores)

		pred = self.mapbackLabels(pred[:,N])
		scores = scores[:,N]

		return (pred, scores)

	def predict(self, X):
		scores = np.zeros((X.shape[0], self.numClasses))
				
		for idx in range(X.shape[0]):
			currNode = self.treeRoot
			while currNode.terminal == False:
				subSpaceX = X[idx, currNode.fSubSpace]
				fVal = np.dot(subSpaceX, currNode.splitModel[:-1]) + currNode.splitModel[-1]

				if fVal < currNode.splitThresh[0]:
					#print "Going left"
					currNode = currNode.leftNode
				else:
					#print "Going Right"
					currNode = currNode.rightNode

			scores[idx,:] = currNode.pDist

		pred = scores.argmax(axis=1)
		pred = self.mapbackLabels(pred)
		return (pred, scores)

	# Tree saving code
	def preOrder(self, node, tree_data):
		if node == None:
			return

		node_data = {}
		node_data['terminal'] = node.terminal

		if node.terminal == True:
			node_data['pDist'] = node.pDist
		else:
			node_data['fSubSpace'] = node.fSubSpace
			node_data['splitModel'] = node.splitModel
			node_data['splitThresh'] = node.splitThresh

		tree_data.append(node_data)

		node.preOrder(node.leftNode, tree_data)
		node.preOrder(node.rightNode, tree_data)

	def dumpt_tree(self, root):
		tree_data = []
		root.preOrder(root, tree_data)

		return tree_data

	def save(self):
		tree_data = {}
		tree_data['labelMap'] = self.labelMap
		tree_data['numClasses'] = self.numClasses
		root = self.treeRoot

		tree_data['tree'] = self.dumpt_tree(root)

		return tree_data

	# Tree reloading code
	def loadLeafNode(self, depth):
		node = RandomizedTree()
		node.dataIDX = None
		node.depth = depth
		node.terminal = True
		node.pDist = None
		node.score = 0.0

		node.fSubSpace = None
		node.splitModel = None
		node.splitThresh = None

		node.leftNode = None
		node.rightNode = None

		return node

	def load_tree(self, node, tree_data):
		node_data = tree_data.pop(0)
		node.terminal = node_data['terminal']

		if node.terminal == True:
			node.pDist = node_data['pDist']
		else:
			node.fSubSpace = node_data['fSubSpace']
			node.splitModel = node_data['splitModel']
			node.splitThresh = node_data['splitThresh']

		if not node.terminal == True:
			node.leftNode = self.loadLeafNode(node.depth + 1)
			node.leftNode = self.load_tree(node.leftNode, tree_data)

			node.rightNode = self.loadLeafNode(node.depth + 1)
			node.rightNode = self.load_tree(node.rightNode, tree_data)

		return node

	def load(self, tree_data):
		root = RandomizedTree()
		root = root.loadLeafNode(1)

		tree = tree_data['tree']
		self.treeRoot = root.load_tree(root, tree)
		self.labelMap = tree_data['labelMap']
		self.numClasses = tree_data['numClasses']
