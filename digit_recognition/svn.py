import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

classifier = svm.SVC(gamma=0.001, C=100)

# Load all but the last 10 data points, so we can use all of these for training.
# Then, we can use the last 10 data points for testing.
Coordinates, target = digits.data[:-10], digits.target[:-10]

# Train the classifier
classifier.fit(Coordinates, target)

def test(index):
    print('Prediction:', index, classifier.predict(digits.data[index]))
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

for i in range(-1,-11,-1):
    test(i)
