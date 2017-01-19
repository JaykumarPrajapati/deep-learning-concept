"""Softmax."""


import numpy as np
import matplotlib.pyplot as plt




def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x),axis=0)

scores = np.array([3.0, 1.0, 0.2])


print(softmax(scores))
print(softmax(scores*10))           # grater score -- more confidance  --- contain pick distrubution
print(softmax(scores*0.5))          # smaller score -- not much confidant  -- contain uniform distribution
plt.plot(softmax(scores*0.05))


# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = 0.5 * np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
plt.plot(x, softmax(scores).T, linewidth=2, label = 'x')
# plt.show()
plt.savefig("./temp.jpg")


# -------------------------------------why subtract contant( C ) from score----------------------

f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
print p         # [  5.75274406e-290   2.39848787e-145   1.00000000e+000]


# ---------------------     SVM   vs. Softmax--------------
'''
    svm:
        for true class=1
                [10,-2,3]       -> loss : 0
                [10,-100,-100]  -> loss : 0

                i.e. does not care for wrong scores once it statishfy margin condition
    softmax:
        not as per svm..... condiering bad scores also ...even it is higher prob. for correct one.

'''