from struct import *
import numpy as np
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
# let Jeff know.
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
        fImages = open('train-images.idx3-ubyte','rb')
        fLabels = open('train-labels.idx1-ubyte','rb')
    else:
        fImages = open('t10k-images.idx3-ubyte','rb')
        fLabels = open('t10k-labels.idx1-ubyte','rb')

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

    T = [] # list of (input, correct label) pairs
    
    for blah in range(0, howMany):
        # get the input from the image file
        x = []
        for i in range(0, rowsIm*colsIm):
            val = unpack('>B',fImages.read(1))[0]
            x.append(val/255.0)

        # get the correct label from the labels file.
        val = unpack('>B',fLabels.read(1))[0]
        y = []
        for i in range(0,10):
            if val==i: y.append(1)
            else: y.append(0)

        T.append((blah,x,y))
            
    fImages.close()
    fLabels.close()

    return T

# this function is not needed to do the training, but just in case you want
# to see what one of the training images looks like.  this will take the
# training data that was produced from the MNSTexample function and write
# it out to a file that you can look at to see what the picture looks like.
# It will write out a separate image for each thing in the training set.
def writeMNISTimage(T):
    # note that you need to have the Python Imaging Library installed to
    # run this function.  If you search for it online, you'll find it.
    import Image
    for i in range(0, len(T)):
        im = Image.new('L',(28,28))
        pixels = im.load()
        for x in range(0,28):
            for y in range(0,28):
                pixels[x,y] = int(T[i][0][x+y*28]*255)
        im.save('mnistFile'+str(i)+'.bmp')

# example of running the last function to write out some of the pictures.
# writeMNISTimage(MNISTexample(0,100,only01=False))


# Gets the averages of all the pictures and writes them to a picture for no reason.


def getAverage(numPics = 60000):
    dataPoints = MNISTexample(0, numPics)
    r = []
    for x in range(0, 784):
        S = 0
        for y in dataPoints:
            S+=y[1][x]
        S/=numPics
        r.append(S)
    writeMNISTimage([(r,0)])



# The meat of the problem. Returns the distance between two handwritten digits.
# Right now it's Euclidean.

def getDistance(V1, V2):
    distance = 0
    for i in range(0, len(V1)):
        distance += (V1[i]-V2[i])**2

    return distance**.5


# returns the Laplacian of the given graph according to the distance function.
def generateLaplacian(T):
    L = np.empty([len(T), len(T)])
    for i in range(0, len(T)):
        for j in range(0, i):
            distance = getDistance(T[i][1], T[j][1])
            L[j][i] = -1*distance
            L[i][j] = -1*distance
        L[i][i] = 0

    for i in range(0, len(T)):
        summ = 0
        for j in range(0, len(T)):
            summ-=L[i][j]
        L[i][i] = summ

    return L


def learn():
    dataPoints = MNISTexample(0, 100)
    laplacian = generateLaplacian(dataPoints)

    print("Done")

learn()