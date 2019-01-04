import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model as lin
import sklearn.neighbors as knn
import operator
import pickle as p

with open('C:\Users\haksh\Documents\CSC411\A2\mnist.pickle\mnist.pickle', 'rb') as f:
    Xtrain, Ytrain, Xtest, Ytest = p.load(f)


def genData(mu0, mu1, Sigma0, Sigma1, N):
    cluster_one = np.random.multivariate_normal(mu0, Sigma0, N)
    target_one = np.zeros((N,1))
    cluster_two = np.random.multivariate_normal(mu1, Sigma1, N)
    target_two = np.ones((N,1))
    cluster_one = np.append(cluster_one, target_one, axis=1)
    cluster_two = np.append(cluster_two, target_two, axis=1)
    X = np.vstack((cluster_one, cluster_two))
    X = sk.utils.shuffle(X)

    return X[:,[0,1]], X[:,2].reshape((2*N, 1))

# create mean and covariance matrices
mu0 = [0, -1]
mu1 = [-1, 1]
Sigma0 = [[2.0,0.5], [0.5,1.0]]
Sigma1 = [[1.0, -1.0] , [-1.0,2.0]]

print ('\nQuestion 1(b)')
X, t = (genData(mu0, mu1, Sigma0, Sigma1, 10000))
print ('\nQuestion 1(c)')
cluster = np.append(X, t, axis=1)
plt.scatter(cluster[np.where(cluster[:,2] == 0)][:,0], cluster[np.where(cluster[:,2] == 0)][:,1], c='red', s=2)
plt.scatter(cluster[np.where(cluster[:,2] == 1)][:,0], cluster[np.where(cluster[:,2] == 1)][:,1], c='blue', s=2)
plt.xlim(-5, 6)
plt.ylim(-5, 6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Question 1(c): sample cluster data (10,000 points per cluster)')
plt.show()

print('\nQuestion 2(a)')
X, t = (genData(mu0, mu1, Sigma0, Sigma1, 1000))
cluster_one = np.append(X, t, axis=1)
cluster_two = np.append(X, t, axis=1)

log_reg = lin.LogisticRegression().fit(X, t)
print ('\nQuestion 2(b)')
print ('Bias term (w0): ' + str(log_reg.intercept_))
print ('Weight vector (w): ' + str(log_reg.coef_))
print ('Mean accuracy: ' + str(log_reg.score(X, t)))

print ('\nQuestion 2(c)')
plt.scatter(cluster_one[np.where(cluster_one[:,2] == 0)][:,0], cluster_one[np.where(cluster_one[:,2] == 0)][:,1], c='red', s=2)
plt.scatter(cluster_two[np.where(cluster_two[:,2] == 1)][:,0], cluster_two[np.where(cluster_two[:,2] == 1)][:,1], c='blue', s=2)
# plot wTx + b
# decision_boundary_x = np.linspace(-5, 6, 2000).reshape(2000, 1)
# decision_boundary_y = 1 / (1 + np.exp(-1 * (np.matmul(decision_boundary_x, log_reg.coef_) + log_reg.intercept_)))
weights = log_reg.coef_.flatten()
decision_boundary_x = X
decision_boundary_y = decision_boundary_x.dot(-1 * weights[0] / weights[1]) - log_reg.intercept_ / weights[1]
plt.plot(decision_boundary_x, decision_boundary_y, c='black')
plt.xlim(-5, 6)
plt.ylim(-5, 6)
plt.title('Question 2(c): training data and decision boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print ('\nQuestion 2(d)')
# plot class 0
plt.scatter(cluster_one[np.where(cluster_one[:,2] == 0)][:,0], cluster_one[np.where(cluster_one[:,2] == 0)][:,1], c='red', s=2)
# plot class 1
plt.scatter(cluster_two[np.where(cluster_two[:,2] == 1)][:,0], cluster_two[np.where(cluster_two[:,2] == 1)][:,1], c='blue', s=2)
decision_boundary_x = X[:,0].reshape((X[:,0].shape[0], 1))
plt.title('Question 2(d): decision boundaries for seven thresholds')
plt.xlabel('x')
plt.ylabel('y')
class_one_false_positives = []
class_two_false_positives = []
base_intercept = log_reg.intercept_
for x in range (-3, 4):
    log_reg.intercept_ = base_intercept - x
    print (log_reg.intercept_)
    predictions = log_reg.predict(X)
    c0_false_p = np.sum(predictions[np.where(t[:,0] == 1)] == 0)
    c1_false_p = np.sum(predictions[np.where(t[:,0] == 0)] == 1)
    print ('For value of t = ' + str(x) + ', there were ' + str(c0_false_p) + ' false negatives of class zero/blue dots.')
    print ('For value of t = ' + str(x) + ', there were ' + str(c1_false_p) + ' false positives of class one/red dots.\n')
    class_one_false_positives.append(c0_false_p)
    class_two_false_positives.append(c1_false_p)
    decision_boundary_y = decision_boundary_x.dot(-1 * weights[0] / weights[1]) - log_reg.intercept_ / weights[1]
    if x < 0:
        plt.plot(decision_boundary_x, decision_boundary_y, c='red')
    elif x > 0:
        plt.plot(decision_boundary_x, decision_boundary_y, c='blue')
    else:
        plt.plot(decision_boundary_x, decision_boundary_y, c='black')
plt.show()

print ('Question 2(e)')
most_fp_t = range(3, -4, -1)[np.where(class_one_false_positives == np.max(class_one_false_positives))[0][0]]
print 'The value of t that gives the most number of blue false positives is: ' + str(most_fp_t)
plt.bar(range(3, -4, -1), class_one_false_positives)
plt.xlabel('t')
plt.ylabel('# of FP')
plt.title('Question 2(e): Explanatory figure')
plt.show()

print('\nQuestion 2(h)')
test_X, test_t = genData(mu0, mu1, Sigma0, Sigma1, 10000)
log_reg.intercept_ = base_intercept - 1
predictions = log_reg.predict(test_X)
true_positives = float(np.sum(predictions[np.where(test_t[:, 0] == 1)] == 1))
false_positives = float(np.sum(predictions[np.where(test_t[:, 0] == 0)] == 1))
false_negatives = float(np.sum(predictions[np.where(test_t[:, 0] == 1)] == 0))
true_negatives = float(np.sum(predictions[np.where(test_t[:,0] == 0)] == 0))
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
print('Predicted positives: ' + str(np.count_nonzero(predictions == 1)))
print('Predicted negatives: ' + str(np.count_nonzero(predictions == 0)))
print('Predicted true positives: ' + str(true_positives))
print('Predicted false positives: ' + str(false_positives))
print('Predicted true negatives: ' + str(true_negatives))
print('Predicted false negatives: ' + str(false_negatives))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))

#plt.scatter(cluster_one[np.where(cluster_one[:,2] == 0)][:,0], cluster_one[np.where(cluster_one[:,2] == 0)][:,1], c='red', s=2)
#plt.scatter(cluster_two[np.where(cluster_two[:,2] == 1)][:,0], cluster_two[np.where(cluster_two[:,2] == 1)][:,1], c='blue', s=2)
#plt.plot(np.linspace(-5, 6, 1000), 1/(1+np.exp(-1 * np.linspace(-5, 6, 1000))))
ind = np.arange(2)
width = 0.35
plt.ylabel('Amount')
plt.xlabel('Classification')
plt.bar(ind[0], true_positives, width)
plt.bar(ind[0], false_positives, width)
plt.bar(ind[1], true_negatives, width)
plt.bar(ind[1], false_negatives, width)
plt.xticks(ind, ('True/False Positives', 'True/False Negatives'))
#plt.legend((ind), ('True Positives', 'False Positives', 'True Negatives', 'False Negatives'))
plt.title('Question 2(h): Explanatory figure')
plt.show()

precision_values = []
recall_values = []
t_values = np.linspace(-5, 6, 1000)

for x in t_values:
    log_reg.intercept_ = base_intercept - x
    predictions = log_reg.predict(test_X)

    true_positives = float(np.sum(predictions[np.where(test_t[:,0] == 1)] == 1))
    false_positives = float(np.sum(predictions[np.where(test_t[:,0] == 0)] == 1))
    false_negatives = float(np.sum(predictions[np.where(test_t[:,0] == 1)] == 0))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    precision_values.append(precision)
    recall_values.append(recall)
#sort the values as otherwise it gives negative AUC
plt.plot(recall_values, precision_values)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Question 2(i): precision/recall curve')
plt.show()

print '\nQuestion 2(j)'
print 'Let there be C total points. Assume all points are classified as blue'
print 'tp is the number of correct guesses, fp is the number of incorrect guesses.'
print 'Thus, since prec = tp/C, prec >= 0.5.'

print('\nQuestion 2(k)')
L = sorted(zip(recall_values, precision_values), key=operator.itemgetter(0))
recall_values, precision_values = zip(*L)
#auc = np.trapz(precision_values, recall_values)
print np.append(np.asarray(np.ediff1d(recall_values)),np.zeros(1))
auc = np.multiply(np.append(np.asarray(np.ediff1d(recall_values)), np.zeros(1)),np.asarray(precision_values)).sum()
print('Area under curve: ' + str(auc))

print '\nQuestion 2(l)'
plt.plot(np.linspace(0, 1, 1000), -1 * np.linspace(0, 1, 1000) + 1)
plt.plot(recall_values, precision_values)
plt.fill_between(np.linspace(0, 1, 1000), -1 * np.linspace(0, 1, 1000) + 1, color='r')
plt.title('Question 2(l): Explanatory figure\nThe area of this triangle is 0.5 => area of curve >= 0.5')
plt.show()

print ('\nQuestion 3')
random_row_indexes = np.random.randint(Xtrain.shape[0], size=36)
random_x = Xtrain[[random_row_indexes],:][0]
random_y = Ytrain[random_row_indexes]

print 'Question 3(a)'
fig, ax = plt.subplots(6,6)
ax = ax.flatten()
count = 0
for row in random_x:
    number = row.reshape(28,28)
    ax[count].imshow(number, cmap='Greys', interpolation='nearest')
    ax[count].axis('off')
    count += 1
plt.suptitle('Question 3(a): 36 random MNIST images.')
plt.show()

print '\nQuestion 3(b)'
clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(Xtrain, Ytrain)
training_accuracy = 100.0 * clf.score(Xtrain, Ytrain)
test_accuracy = 100.0 * clf.score(Xtest, Ytest)
print 'Training accuracy: ' + str(training_accuracy)
print 'Test accuracy: ' + str(test_accuracy)

print '\nQuestion 3(c)'
#print 'Training Accuracy    | Test Accuracy     | k value'
#training_accuracies = np.zeros(20)
test_accuracies = np.zeros(20)
for k in range(1, 21):
    knn_clf = knn.KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn_clf.fit(Xtrain, Ytrain)
    #knn_train_acc = knn_clf.score(Xtrain, Ytrain)
    knn_test_acc = knn_clf.score(Xtest, Ytest)
    #training_accuracies[k-1] = knn_train_acc
    test_accuracies[k-1] = 100.0 * knn_test_acc
    #print str(knn_train_acc) + '\t| ' + str(knn_test_acc) + '\t| ' + str(k)
best_k = np.where(test_accuracies == max(test_accuracies))[0] + 1
print 'Best K value: ' + str(best_k)
plt.plot(range(1, 21), test_accuracies)
plt.title('3(c): KNN test accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()

print ('\nQuestion 5')
def softmax1(z):
    #compute the sum of the exp of the components just once, no need to do it multiple times
    #compute the softmax function
    return np.exp(z) / np.sum(np.exp(z))

def softmax2(z):
    #z = z - np.max(z)
    y = np.asarray(np.exp(z-np.max(z)) / np.sum(np.exp(z - np.max(z))))
    logy = np.asarray(z - np.max(z) - np.log(np.sum(np.exp(z - np.max(z)))))
    return y, logy

print '\nQuestion 5(c)'
print softmax2((0,0))
print softmax2((1000,0))
print softmax2((-1000,0))

print ('\nQuestion 6')
print ('\nQuestion 6(a)')
K = 10
weights = 0.1 * np.random.randn(Xtrain.shape[1], K)
#bias_terms = np.reshape(, (Xtrain.shape[0], 1))
bias_terms = np.zeros(K)
#weights = np.hstack((bias_terms, weights))
#weights[:,0] = bias_terms
#[:,:-1]
#create arrays, 5000 epochs / 10 epoch cycles = 500 entries
training_losses = np.zeros(500)
test_losses = np.zeros(500)
training_accuracies = np.zeros(500)
test_accuracies = np.zeros(500)
#convert Ytrain and Ytest to 1 of K
t_train = np.zeros((Ytrain.shape[0], K))
t_train[np.arange(Ytrain.shape[0]), Ytrain] = 1
t_test = np.zeros((Ytest.shape[0], K))
t_test[np.arange(Ytest.shape[0]), Ytest] = 1

print ('\nQuestion 6(b)')
lrate = 1
for i in range(5000):

    predictions = np.matmul(Xtrain, weights) + bias_terms
    m = np.max(predictions)
    y = np.asarray(np.exp(predictions - m) / np.sum(np.exp(predictions - m), axis=1).reshape(Xtrain.shape[0], 1))
    logy = np.asarray(predictions - m - np.log(np.sum(np.exp(predictions - m), axis=1)).reshape(Xtrain.shape[0], 1))
    #y.reshape(y.shape[0], 1)
    #update weights using gradient descent
    diff = y - t_train
    derived_loss = np.dot(Xtrain.T, diff) / Xtrain.shape[0]
    weights -= lrate * derived_loss
    bias_terms = bias_terms - lrate * np.sum(diff) / Xtrain.shape[0]

    if i % 10 == 0:
        test_predictions = np.dot(Xtest, weights) + bias_terms
        m_t = np.max(test_predictions)
        y_test = np.asarray(np.exp(test_predictions - m_t) / np.sum(np.exp(test_predictions - m_t), axis=1).reshape(Xtest.shape[0], 1))
        logy_test = np.asarray(test_predictions - m_t - np.log(np.sum(np.exp(test_predictions - m_t), axis=1)).reshape(Xtest.shape[0], 1))
        indx = int(i/10)
        #print logy
        training_accuracy = 100 * np.mean(np.argmax(y, axis=1) == np.argmax(t_train, axis=1))
        test_accuracy = 100 * np.mean(np.argmax(y_test, axis=1) == np.argmax(t_test, axis=1))
        training_accuracies[indx] = training_accuracy
        test_accuracies[indx] = test_accuracy

        #print np.multiply(t_train, logy)
        #print np.sum(np.multiply(t_train, logy), axis=1)
        training_error = -1 * np.sum(np.sum(np.multiply(t_train, logy), axis=1)) / Xtrain.shape[0]
        test_error = -1 * np.sum(np.sum(np.multiply(t_test, logy_test), axis=1)) / Xtest.shape[0]
        training_losses[indx] = training_error
        test_losses[indx] = test_error

        print 'i value: ' + str(i)
        print training_accuracy
        print test_accuracy
        print training_losses[indx]
        print test_losses[indx]

print '\nQuestion 6(d)'
print 'Learning rate: ' + str(lrate)
print 'Final Training Loss: ' + str(training_losses[-1])
print 'Final Test Loss: ' + str(test_losses[-1])
print 'Final Training Accuracy: ' + str(training_accuracies[-1])
print 'Final Test Accuracy: ' + str(test_accuracies[-1])

plt.semilogx(np.linspace(0, 5000, 500), training_accuracies, c='orange')
plt.semilogx(np.linspace(0, 5000, 500), test_accuracies, c='blue')
plt.xlabel('Epoch')
plt.title('Question 6(e): training and test accuracy for batch gradient descent.')
plt.ylabel('Accuracy')
plt.show()

plt.semilogx(np.linspace(0, 5000, 500), training_losses, c='red')
plt.semilogx(np.linspace(0, 5000, 500), test_losses, c='blue')
plt.title('Question 6(f): training and test loss for batch gradient descent.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(np.linspace(3000, 5000, 200), training_accuracies[-200:])
plt.title('Question 6(g): training accuracy for last 2000 epochs of bgd')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(np.linspace(3000, 5000, 200), training_losses[-200:])
plt.title('Question 6(h): training loss for last 2000 epochs of bgd')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

print '\nQuestion 7'

training_losses = np.zeros(500)
test_losses = np.zeros(500)
training_accuracies = np.zeros(500)
test_accuracies = np.zeros(500)

batchSize = 1000
epochs = 500
lrate = 1

weights = 0.1 * np.random.randn(Xtrain.shape[1], 10)
bias_terms = np.zeros(10)

for epoch in range(epochs):
    print 'Epoch: ' + str(epoch)
    # shuffle train and test data in order
    combined_train_dataset = np.hstack((Xtrain, Ytrain.reshape(Ytrain.shape[0], 1)))
    np.random.shuffle(combined_train_dataset)
    combined_test_dataset = np.hstack((Xtest, Ytest.reshape(Ytest.shape[0], 1)))
    np.random.shuffle(combined_test_dataset)

    # break apart the shuffled dataset into their X and Y components
    shuffledXtrain = combined_train_dataset[:, :-1]
    shuffledYtrain = combined_train_dataset[:, -1]
    shuffledXtest = combined_test_dataset[:, :-1]
    shuffledYtest = combined_test_dataset[:, -1]

    for i in range(Xtrain.shape[0] // batchSize):
        start_index = i * batchSize
        end_index = start_index + batchSize
        batchXtrain = shuffledXtrain[start_index:end_index,]
        batchYtrain = shuffledYtrain[start_index:end_index,]
        batchXtest = shuffledXtest[start_index:end_index,]
        batchYtest = shuffledYtest[start_index:end_index,]

        t_train = np.zeros((batchYtrain.shape[0], 10), dtype=float)
        t_train[np.arange(batchYtrain.shape[0]), batchYtrain.astype(dtype=np.int8)] = 1.0
        t_test = np.zeros((batchYtest.shape[0], 10), dtype=float)
        t_test[np.arange(batchYtest.shape[0]), batchYtest.astype(dtype=np.int8)] = 1.0

        predictions = np.matmul(batchXtrain, weights) + bias_terms
        m = np.max(predictions)
        y = np.asarray(np.exp(predictions - m) / np.sum(np.exp(predictions - m), axis=1).reshape(batchXtrain.shape[0], 1))
        logy = np.asarray(predictions - m - np.log(np.sum(np.exp(predictions - m), axis=1)).reshape(batchXtrain.shape[0], 1))
        # y.reshape(y.shape[0], 1)
        # update weights using gradient descent
        diff = y - t_train
        derived_loss = np.dot(batchXtrain.T, diff) / batchXtrain.shape[0]
        weights -= lrate * derived_loss
        bias_terms = bias_terms - lrate * np.sum(diff) / batchXtrain.shape[0]
        # at the end of every epoch
        # this could be done outside the batch loop, but it seemed cleaner this way to me,
        # since otherwise we'd have to reinstantiate the arrays
        if i == (Xtest.shape[0] // batchSize) - 1:

            t_train = np.zeros((Ytrain.shape[0], 10))
            t_train[np.arange(Ytrain.shape[0]), Ytrain] = 1
            t_test = np.zeros((Ytest.shape[0], 10))
            t_test[np.arange(Ytest.shape[0]), Ytest] = 1

            # calculate predictions for entire dataset
            predictions = np.dot(Xtrain, weights) + bias_terms
            m = np.max(predictions)
            y = np.asarray(
                np.exp(predictions - m) / np.sum(np.exp(predictions - m), axis=1).reshape(Xtrain.shape[0], 1))
            logy = np.asarray(
                predictions - m - np.log(np.sum(np.exp(predictions - m), axis=1)).reshape(Xtrain.shape[0], 1))

            # calculate predictions for entire test dataset
            test_predictions = np.dot(Xtest, weights) + bias_terms
            m_t = np.max(test_predictions)
            y_test = np.asarray(
                np.exp(test_predictions - m_t) / np.sum(np.exp(test_predictions - m_t), axis=1).reshape(Xtest.shape[0], 1))
            logy_test = np.asarray(
                test_predictions - m_t - np.log(np.sum(np.exp(test_predictions - m_t), axis=1)).reshape(Xtest.shape[0], 1))

            training_accuracy = 100.0 * np.sum(np.argmax(y, axis=1) == np.argmax(t_train, axis=1)) / Xtrain.shape[0]
            test_accuracy = 100.0 * np.sum(np.argmax(y_test, axis=1) == np.argmax(t_test, axis=1)) / Xtest.shape[0]
            training_accuracies[epoch] = training_accuracy
            test_accuracies[epoch] = test_accuracy

            # print np.multiply(t_train, logy)
            # print np.sum(np.multiply(t_train, logy), axis=1)
            training_error = -1 * np.sum(np.sum(np.multiply(t_train, logy), axis=1)) / Xtrain.shape[0]
            test_error = -1 * np.sum(np.sum(np.multiply(t_test, logy_test), axis=1)) / Xtest.shape[0]
            training_losses[epoch] = training_error
            test_losses[epoch] = test_error

            print training_losses[epoch]
            print test_losses[epoch]
            print training_accuracies[epoch]
            print test_accuracies[epoch]

print '\nQuestion 7(d)'
print 'Learning rate: ' + str(lrate)
print 'Final Training Loss: ' + str(training_losses[-1])
print 'Final Test Loss: ' + str(test_losses[-1])
print 'Final Training Accuracy: ' + str(training_accuracies[-1])
print 'Final Test Accuracy: ' + str(test_accuracies[-1])

plt.semilogx(np.linspace(0, 500, 500), training_accuracies, c='orange')
plt.semilogx(np.linspace(0, 500, 500), test_accuracies, c='blue')
plt.ylim(80, 100)
plt.xlabel('Epoch')
plt.title('Question 7(e): training and test accuracy for stochastic gradient descent.')
plt.ylabel('Accuracy')
plt.show()

plt.semilogx(np.linspace(0, 500, 500), training_losses, c='orange')
plt.semilogx(np.linspace(0, 500, 500), test_losses, c='blue')
plt.ylim(0.2, 1)
plt.title('Question 7(f): training and test loss for stochastic gradient descent.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

min_epoch = np.where(np.min(test_losses) == test_losses)[0][0]
print '\nQuestion 7(g)'
print 'Minimal test loss occurs at epoch #' + str(min_epoch+1)
print 'Training Accuracy at epoch #' + str(min_epoch+1) + ': ' + str(training_accuracies[min_epoch])
print 'Test Accuracy at epoch #' + str(min_epoch+1) + ': ' + str(test_accuracies[min_epoch])
print 'Training Loss at epoch #' + str(min_epoch+1) + ': ' + str(training_losses[min_epoch])
print 'Test Loss at epoch #' + str(min_epoch+1) + ': ' + str(test_losses[min_epoch])

