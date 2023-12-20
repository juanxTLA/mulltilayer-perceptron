import numpy as np

class MLP:
    '''
        Constructor:
            layers -> integer containing number of layers including output and input layer
            neurons -> list containing the number of neurons per layer including input and output
            activations -> list that contains a string naming the activation function
                            to be used in every layer
            initMethod -> string for initialization method to be used
            
            LENGTH OF NEURONS LIST MUST EQUAL NUMBER OF LAYERS
            LENGTH OF ACTIVATIONS LIST MUST BE OF LAYERS - 1
    '''
    
    def __init__(self, layers : int, neurons : list, activations : list, initMethod = "zeros") -> None:

        self.nLayers = layers
        self.architecture = neurons #useful when saving data into file
        self.outNeurons = neurons[-1]
        
        self.wValues = [] #list of 2darrays for w in each hidden layer
        self.bValues = [] #list of arrays for b in each hidden layer

        self.predictions = []

        self.activations = activations

        #initialize weights and biases
        if(initMethod == "random"):
            for l in range(1,layers):
                self.wValues.append(np.random.randn(neurons[l-1], neurons[l]))
                self.bValues.append(np.zeros((1,neurons[l])))

        elif(initMethod == "zeros"):
            for l in range(1,self.nLayers):
                self.wValues.append(np.zeros((neurons[l-1], neurons[l])))
                self.bValues.append(np.zeros((1,neurons[l])))
        
        #if any other string is input we assume it is a filename, and thus we will load the network parameters
        else:
            self.loadNetwork(initMethod) 

    #need to do the activation function for each batch
    def relu(self,x):
        return np.maximum(0,x)

    def sigmoid(self,x):
        x = np.clip(x, -500, 500)
        x = 1/(1+np.exp(-x))
        return x
    
    def softmax(self, x):
        x = np.clip(x, -500, 500)
        
        #this block is trying to fix an issue when composing the softmax after loading the neural network
        if(x.ndim == 1):
            exps = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exps / np.sum(exps, axis=0, keepdims=True)

        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    
    def getDerivative(self, x, function : str):
        if(function == "relu"):
            x = np.where(x > 0, 1, 0)

        #derivative for sigmoid and softmax is f'(x) = f(x) * (1-f(x))
        elif(function == "sigmoid"):
            x = self.sigmoid(x) * (1 - self.sigmoid(x))

        elif(function == "softmax"):
            x = self.softmax(x) * (1 - self.softmax(x))
        
        return x
    
    # sum[i!=j](max(1+yi - yj, 0)) j is the index of the actual value
    #compute hinge loss for a specific batch
    def hingeLoss(self, y : list, yTrue : list):
        sum = 0

        for pred, label in zip(y,yTrue):
            j = np.argmax(label)
            yj = pred[j]
            np.delete(pred, j)
            
            for i in pred:
                sum += max(1+i-yj, 0)
        
        error = sum/len(yTrue)
        
        return error

    #x is array of inputs (vectorized)
    #data needs to be arranged before-hand
    def forwardPass(self, x):

        aValues = [x]
        zValues = []
        
        #now we can do the rest
        for l in range(self.nLayers - 1):
            zValues.append(aValues[l] @ self.wValues[l] + self.bValues[l])
                    
            if(self.activations[l] == 'relu'):
                aValues.append(self.relu(zValues[l]))        

            elif(self.activations[l] == 'sigmoid'):
                aValues.append(self.sigmoid(zValues[l]))      

            elif(self.activations[l] == 'softmax'):
                aValues.append(self.softmax(zValues[l]))      

        return aValues

    def crossEntropyLoss(self, a, y):
        epsilon = 1e-15 #used to prevent errors
        return np.sum(np.nan_to_num(-y*np.log(a+epsilon)-(1-y+epsilon)*np.log(1-a+epsilon)))

    #this will calculate the overall loss, which means copmputing the loss for each minibatch    
    def getLoss(self, yTrue : list, yComp : list, lossFunc : str):
        cumError = 0

        if(lossFunc == "cross-entropy"):
            for i in range(len(yTrue)):
                cumError += self.crossEntropyLoss(yComp[i], yTrue[i])
            cumError /= len(yTrue)
        
        elif(lossFunc == "hinge-loss"):
            cumError = self.hingeLoss(yComp, yTrue)
        
        return cumError
    
    def backwardPass(self, aValues, yTrue, learningRate = 0.05, lossFunc = "cross-entropy"):
        #error = get the loss from the current weights and outputs
        error = self.getLoss(yTrue, aValues[-1], lossFunc)
        #we first do the derivative of the output layer to compute the delta of the error

        dw = self.wValues.copy()
        db = self.bValues.copy()

        #output (commented out for the second trun using hinge loss)
        '''
        if(lossFunc == "cross-entropy"):
            dz = (aValues[-1] - yTrue) / yTrue.shape[0]

        elif(lossFunc == "hinge-loss"):
            dz = np.where(yTrue * aValues[-1] >= 1, 0, -yTrue)        
        '''
        dz = (aValues[-1] - yTrue) / yTrue.shape[0]
        dw[-1] = aValues[-2].T @ dz
        db[-1] = np.sum(dz[-1])

        self.wValues[-1] -= learningRate * dw[-1]
        self.bValues[-1] -= learningRate * db[-1]

        #then we back propagate the error for the hidden layers
        for l in reversed(range(2, self.nLayers - 1)):
            localDeriv = self.getDerivative(aValues[l], self.activations[l])
            dz = (dz @ self.wValues[l].T) * localDeriv
            dw[l] = aValues[l-1].T @ dz
            db[l] = np.sum(dz)
            self.wValues[l-1] -= learningRate * dw[l]
            self.bValues[l-1] -= learningRate * db[l]

        return error
    
    #function to train the network
    #returns two dictionaries with the correspondent error and accuracy for both test and train datasets
    def train(self, x, yTrue, testImage, testLabel, epochs = 100, learningRate = 0.05, batchNum = 1, costFunc = "cross-entropy"):
        n = len(yTrue) #yTrue and x shall be the same length (60000 for MNIST)
        
        batchSize = int(n/batchNum)
        imageBatches = []
        labelBatches = []

        for i in range(0, batchSize):
            imageBatches.append(np.vstack((x[i:i+batchSize])))
            labelBatches.append(np.vstack((yTrue[i:i+batchSize])))

        aValues = []
        error = {'t' : [], 'v' : []}
        accuracy = {'t' : [], 'v' : []}

        #we need to get the data into batches, both inputs and labels
        #the x input is already linearized

        for i in range(epochs):
            cumError = 0
            for image,label in zip(imageBatches, labelBatches):
                aValues = self.forwardPass(image)
                cumError += self.backwardPass(aValues, label, learningRate, costFunc)
            cumError /= batchSize

            validAcc = self.getAccuracy(testImage, testLabel)
            trainAcc = self.getAccuracy(x, yTrue)

            accuracy['t'].append(trainAcc)
            accuracy['v'].append(validAcc)

            #get the overall error from the validation images
            valForw = []
            for image in testImage:
                valForw.append(self.forwardPass(image)[-1])

            error['t'].append(cumError)
            
            #uncomment if user wants to track the process
            print(i)
        
        return error, accuracy
    
    #after a forward pass and a label
    #option != 0: label is passed for accuracy comparison
    #option == 0: no label is passed an prediction is made
    def predict(self, x, yTrue = [], opt = 0):
        aValues = self.forwardPass(x)
        if opt != 0:
            return (np.argmax(aValues[-1]), np.argmax(yTrue))
        else:
            return np.argmax(aValues[-1])
    
    #store the information of the neural networ in a txt file. This file can be interpreted by this same API
    def saveToFile(self, filename : str):
        with open(filename, 'w') as file:
            #we will first output the neurons list
            for n in self.architecture:
                file.write(str(n) + " ")

            file.write("\n")

            for n in self.activations:
                file.write(n + " ")

            #new line
            file.write("\n")
            #for each weight matrix in wValues we will ouput a line with the dimensions
            for m in self.wValues:
                shape = m.shape
                file.write(str(shape[0]) + " " + str(shape[1]) + "\n")
                #then we ouput the actual weight values
                for row in range(shape[0]):
                    for col in range(shape[1]):
                        file.write(str(m[row][col]) + " ")
                    
                    file.write("\n")

            for b in self.bValues:
                shape = b.shape
                file.write(str(shape[1]) + "\n")
                for i in range(shape[1]):
                    file.write(str(b[0][i]) + ' ')

                file.write("\n")
            
            file.close()

    def loadNetwork(self, filename: str):
        with open(filename, "r") as file:
            next(file)
            next(file)

            for i in range(self.nLayers - 1): # we will have layers - 1 weight matrices
                #get dimensions of matrix
                line = file.readline().strip().split(" ")
                row = int(line[0])
                col = int(line[1])

                temp = np.zeros((row, col)) #array we will populate
                tempRow = 0

                
                for line in file:
                    line = line.strip().split(" ")
                    for c in range(col):
                        temp[tempRow][c] = float(line[c])
                    tempRow += 1
                    if(tempRow == row):
                        break
                
                self.wValues.append(temp)

            for i in range(self.nLayers - 1): # we will have layers - 1 bias arrays
                #get dimensions of matrix
                line = file.readline()
                row = int(line.strip().split(" ")[0])

                temp = np.zeros(row) #array we will populate
                tempRow = 0

                line = file.readline().strip().split(" ")
                for c in range(len(line)):
                    temp[c] = float(line[c])
                    
                self.bValues.append(temp)

    #given a set of inputs and a set of labels, we will return the accuracy of the network
    def getAccuracy(self, x, xTrue):
        countGood = 0
        total = len(x)
        for image, label in zip(x, xTrue):
            res = self.predict(image, label, 1)
            if(res[0] == res[1]):
                countGood += 1

        return countGood/total
            
    #this function will take a list of new activation functions and change it in the network
    #new list of functions must equal number of layers -1
    def changeActivationFunctions(self, new : list):
        if len(new) != self.nLayers -1: #if the list is not valid return false to show something went wrong
            return False

        self.activations = new

        return True #will return true if the change was successful            
                    

    def altTrain(labels, images, epochs, lr):


