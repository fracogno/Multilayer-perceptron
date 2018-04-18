import numpy as np

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def derivative_sigmoid(x):
	return x * (1 - x)


# [HOURS SLEPT, HOURS STUDIED]
X = np.array([[7,1], [6,5], [4,7], [4,3]])

# [EXAM PASSED IF 1]
Y = np.array([[0], [1], [1], [0]])

N, C = X.shape
H = 16	#dimension hidden layer

assert(N == Y.shape[0])

W1 = np.random.uniform(size=(C, H))
W2 = np.random.uniform(size=(H, 1))

iterations = 10000
lr = 0.1	#learning rate
for i in range(iterations):

	#FORWARD PATH
	Y1 = np.dot(X, W1)
	Y2 = sigmoid(Y1)
	Y3 = np.dot(Y2, W2)
	Y4 = sigmoid(Y3)

	if i == (iterations-1):
		print("Percentages: ")
		print(np.around(Y4, 4))	#SHOW FINAL PERCENTAGE OF BEING 1 or 0

	if i % 1000 == 0:
		E = (1/2) * np.sum((Y - Y4)**2)		#I WANT TO MINIMIZE THIS ERROR FUNCTION
		print(E)

	#BACKWARD PATH  dE/d?
	dY4 = - (Y - Y4)
	dY3 = dY4 * derivative_sigmoid(Y4)
	dW2 = np.dot(dY3.T, Y2).T
	dY2 = np.dot(dY3, W2.T)
	dY1 = dY2 * derivative_sigmoid(Y2)
	dW1 = np.dot(dY1.T, X).T

	#GRADIENT DESCENT
	W2 = W2 - lr * dW2
	W1 = W1 - lr * dW1