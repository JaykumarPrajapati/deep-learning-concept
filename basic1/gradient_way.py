import numpy as np

'''
    gradient backward calculation for matrix
'''
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)    #(5,10)(10,3) = (5,3)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D        (5,3)
dW = dD.dot(X.T) #.T gives the transpose of the matrix   (5,3) (10,3)'= (5,10)
dX = W.T.dot(dD)                                        # (5,10)' (5,3) = (10,3)


