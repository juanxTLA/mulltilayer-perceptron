The saved graph corresponds to the neural network in the jupyter cell.
There are two drivers for the two different networks
The neural networks I used are saved in the txt files

Accuracy is not the best it can be, probably with more work on it and some enhancements it could be brought up to 90 or more
The most I got was 84%, however I did not save that one, and it took too long as to run it again. In addition it used relu, sigmoid,
and softmax in different layers which I don't know if it is advisable

PROBLEMS
    -There is an issue when trying to get a prediction from a loaded network.
    The issue arises in line 56 of mlp.py and is a limitation of the implementation
    I have tried using the debugger and all the values seem to be loading into the network just fine,
    but the error keeps showing. What happens is that the dimension of the input to the softmax is transposed
    whenever it is a loaded network. I know we were not supposed to implement softmax but since it is
    a better version of sigmoid I took the liberty of implementing it. And for the training and test of the
    network it worked.

    -An issue I found is that the bias for the first layer for some reason the bias is 0, which means that:
    in this snippet for the backward pass (line 157 in mlp.py):

        localDeriv = self.getDerivative(aValues[l], self.activations[l])
        dz = (dz @ self.wValues[l].T) * localDeriv
        dw[l] = aValues[l-1].T @ dz
        db[l] = np.sum(dz)
        self.wValues[l-1] -= learningRate * dw[l]
        self.bValues[l-1] -= learningRate * db[l]

        for the last iteration db[l] = 0, which can only happen if dz is all 0s and ultimately if the
        derivative for the first layer's activation function is also all 0s

        this can dramatically affect the accuracy of the network, although in previous trainings the highest it got was
        to an 84% on the test set (using softmax)
    
    -Hinge-loss is implemented correctly, but I am thinking the derivative of the loss (which I got from internet) maybe is wrong so the graph wont show any meaninfull. I commented the code out and made a run with the derivative being yhat- yPredicted to see if it has more sense and that is what I put in the second .html file