from struct import *
import numpy as np
import sklearn
from numpy import linalg as LA
import random
# This function reads data from the MNIST handwriting files.  To use this
# you need to download the MNIST files from
#    http://yann.lecun.com/exdb/mnist/
# The data format is described towards the bottom of the page, but this
# function MNISTexample takes care of reading it for you.  It will return
# a list of labeled examples.  Each image in the training files are
# 28x28 grayscale pictures, so the input for each example will have
# 28*28=784 different inputs.  In the function, I have scaled these values
# so they are each between 0.0 and 1.0.  Each of the images could be any
# of the digits 0, 1, ..., 9.  So we should make a neural net that has 10
# different output neurons, each one testing whether the input corresponds
# to one of the digits.  In the examples that are returned by MNISTexample,
# y is a list of length 10, with a 1 in the spot for the correct digit, and
# 0's elsewhere.
#
# NOTE: you should try running the MNISTexample function to get
# just a single example, like MNISTexample(0,1), to make sure it looks
# right.  The header information should look like what they talked about
# on the website, and you can print those values in the function below
# to make sure it looks like it is working.  If it seems messed up,
# let Geoff know.
#
# Inputs to this function...
#
# bTrain says whether to read from the train file for from the test file.
# For the test file, they made sure the examples came from different people
# than were used for producing the training file.
#
# The train file has 60,000 examples, and the test has 10,000.
# startN says which example to start reading from in the file.
# howMany says how many exmaples to read from that point.
#
# only01 is set to True to only return examples where the correct answer
# is 0 or 1.  This makes the task simpler because we're only trying to
# distinguish between two things instead of 10, meaning we won't need to
# train as long to start getting good results.

def MNISTexample(startN,howMany,bTrain=True):
    if bTrain:
        fImages = open('train-images-idx3-ubyte','rb')
        fLabels = open('train-labels-idx1-ubyte','rb')
    else:
        fImages = open('t10k-images-idx3-ubyte','rb')
        fLabels = open('t10k-labels-idx1-ubyte','rb')

    # read the header information in the images file.
    s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
    mnIm = unpack('>I',s1)[0]
    numIm = unpack('>I',s2)[0]
    rowsIm = unpack('>I',s3)[0]
    colsIm = unpack('>I',s4)[0]
    # seek to the image we want to start on
    fImages.seek(16+startN*rowsIm*colsIm)

    # read the header information in the labels file and seek to position
    # in the file for the image we want to start on.
    mnL = unpack('>I',fLabels.read(4))[0]
    numL = unpack('>I',fLabels.read(4))[0]
    fLabels.seek(8+startN)

    digitInfoList = [] # list of (input, correct label) pairs
    
    for blah in range(0, howMany):
        # get the input from the image file
        x = []
        for i in range(0, rowsIm*colsIm):
            val = unpack('>B',fImages.read(1))[0]
            x.append(val/255.0)

        # get the correct label from the labels file.
        val = unpack('>B',fLabels.read(1))[0]
        """
        y = []
        for i in range(0,10):
            if val==i: y.append(1)
            else: y.append(0)
        """

        digitInfoList.append((x,val))
            
    fImages.close()
    fLabels.close()

    return digitInfoList

# this function is not needed to do the training, but just in case you want
# to see what one of the training images looks like.  this will take the
# training data that was produced from the MNSTexample function and write
# it out to a file that you can look at to see what the picture looks like.
# It will write out a separate image for each thing in the training set.
def writeMNISTimage(digitInfoList):
    # note that you need to have the Python Imaging Library installed to
    # run this function.  If you search for it online, you'll find it.
    import Image
    for i in range(0, len(digitInfoList)):
        im = Image.new('L',(28,28))
        pixels = im.load()
        for x in range(0,28):
            for y in range(0,28):
                pixels[x,y] = int(digitInfoList[i][0][x+y*28]*255)
        im.save('mnistFile'+str(i)+'.bmp')

# example of running the last function to write out some of the pictures.
# writeMNISTimage(MNISTexample(0,100,only01=False))

# gets the index of 1 from a list

def get_index(L):
    for i in range(0, 10):
        if L[i] == 1:
            return i

# generates the average of all the pixels in an image for a given
# digit. The corresponding image for each digit is in the repo.

def layerAllImages(digitInfoList):
    # creates clustered versions of images of digits from 1 to 10.
    A = []
    for i in range(0, 10):
        x = []
    A.append(x)
    for k in range(0, 5000):
        p = digitInfoList[k] #the k-th digit
    l = p[0] #the image exactly
    m = get_index(p[1]) #the digit
    A[m].append(l)
    average_image_vector = []
    for i in range(0, 10):
        x = []
        average_image_vector.append(x)
    for i in range(0, 10):
        for k in range(0, 784):
            pixel_weight = 0
            for j in range(0, len(A[i])):
                pixel_weight += A[i][j][k]
        pixel_weight = pixel_weight / len(A[i])
        average_image_vector[i].append(pixel_weight)
    return_value = []
    for i in range(0, 10):
        x = (average_image_vector[i], i)
        return_value.append(x)
    return return_value

# Gets the averages of all the pictures and writes
# them to a picture for no reason.

def getAverage(numPics = 60000):
    dataPoints = MNISTexample(0, numPics)
    r = []
    for x in range(0, 784):
        S = 0
        for y in dataPoints:
            S+=y[0][x]
        S/=numPics
        r.append(S)
    writeMNISTimage([(r,0)])



# The meat of the problem. Returns the distance between two handwritten digits.
# Right now it's Euclidean squared.

def euclidDistance(V1, V2):
    distance = 0
    for i in range(0, len(V1)):
        distance += (V1[i]-V2[i])**2

    return distance


# returns the Laplacian of the given graph according to the distance function.
def generateLaplacian(digitInfoList):
    L = np.empty([len(digitInfoList), len(digitInfoList)])
    for i in range(0, len(digitInfoList)):
        for j in range(0, i):
            distance = np.exp(-0.0005*(euclidDistance(digitInfoList[i][0], digitInfoList[j][0])))
            L[j][i] = -1*distance
            L[i][j] = -1*distance
        L[i][i] = 0

    for i in range(0, len(digitInfoList)):
        summ = 0
        for j in range(0, len(digitInfoList)):
            summ-=L[i][j]
        L[i][i] = summ

    return L


# k-means should be applied on eigencollection.
def kMeansIteration(points, clusterCenters, shouldReturn):
    # Partition the points based on which center each is closest to.
    # pointPartitions is a list of each of the new partitions, which themselves are lists.
    #print("Points:", points)
    #print("ClusterCenters", clusterCenters)
    pointPartitions = [[] for y in range(0, len(clusterCenters))] 
    # This loop iterates over every point
    for pointIndex in range(0, len(points)):
        minDistance = euclidDistance(points[pointIndex], clusterCenters[0])
        minIndex = 0
        # This loop finds the nearest cluster center to the point at pointIndex
        for centerIndex in range(1, len(clusterCenters)):
            distance = euclidDistance(points[pointIndex], clusterCenters[centerIndex])
            if(distance<minDistance):
                minDistance = distance
                minIndex = centerIndex
        pointPartitions[minIndex].append(pointIndex)
    #print(pointPartitions)
    if(shouldReturn):
        return pointPartitions

    # update the cluster centers
    dimension = len(clusterCenters[0])
    for centerIndex in range(0, len(clusterCenters)):
        if(len(pointPartitions[centerIndex])==0):
            continue
        newCenter = [0 for y in range(0, dimension)]
        for coordinate in range(0, dimension):
            for pointIndex in pointPartitions[centerIndex]:
                newCenter[coordinate] += points[pointIndex][coordinate]
            newCenter[coordinate] /= len(pointPartitions[centerIndex])
        clusterCenters[centerIndex] = newCenter

def kMeans(points, numIterations, numClusters, randRange):
    #points = np.matrix.transpose(eigenVectors)
    dimension = len(points[0])
    #print(dimension)
    #print(len(points))
    #randomly generates cluster centers
    clusterCenters = [[random.uniform(randRange[0], randRange[1]) for y in range(0, dimension)] for x in range(0, numClusters)]

    for x in range(0, numIterations):
        #print(clusterCenters)
        kMeansIteration(points, clusterCenters, False)
    return kMeansIteration(points, clusterCenters, True)

"""
# the idea is to subsample the data into many parts and use
# each subsample to create a prediction model
def generateSubsamples(digitInfoList, size):
    dataSize = len(digitInfoList)
    numberOfSamples = len(digitInfoList) / size
    laplaceList = []
    for i in range(0, numberOfSamples):
        print i
        laplace = generateLaplacian(digitInfoList[size*i : size*(i+1)])
        laplaceList.append(laplace)
    return laplaceList
"""

# This is the algorithm for spectral clustering
# digitInfoList is the data sample to learn from.

def spectralClustering(digitInfoList, numClusters):
    laplacian = generateLaplacian(digitInfoList)

    # this is the list of all subsampled Laplacians
    w, v = LA.eig(laplacian)
    idx = w.argsort()
    v = v[:, idx]
    # w is the list of eigenvalues of the matrix.
    # and v comprises of the corresponding
    # eigenvectors.
    # We now convert matrix v into a 2D list for easier handling later.
    v = v.tolist()
    # Only take the first eigenvectors, depending on numClusters
    for i in range(0, len(v)):
        v[i] = v[i][:numClusters]
    # Now, do KMeans on this list
    clusters = kMeans(v, 50, numClusters, (-.15, .15))

    #Convert every index of the cluster back to the proper digit
    for cluster in clusters:
        for index in range(0, len(cluster)):
            cluster[index] = digitInfoList[cluster[index]][1]
    return clusters

def learn(sampleSize, numSamples):

    dataPoints = MNISTexample(0, sampleSize*numSamples)
    #print(dataPoints)
    for i in range(0, numSamples):
        print(spectralClustering(dataPoints[i*sampleSize:(i+1)*sampleSize], 20))


learn(300, 1)