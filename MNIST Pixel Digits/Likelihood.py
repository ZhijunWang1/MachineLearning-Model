'''
Starter code provided by professor 	Roger Grosse
'''
import math

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import numpy.linalg as lg


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        a = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(a, axis=0)
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(10):
        a = data.get_digits_by_label(train_data, train_labels, i)
        b = np.mean(a, axis=0)
        dim1, dim2 = a.shape
        matrixsum = np.zeros((64, 64))
        diagonal = np.zeros((64, 64))
        np.fill_diagonal(diagonal, 0.01)
        for j in range(dim1):
            sample = (a[j] - b).reshape(1, 64)
            # reshape to conduct dot product
            matrixsum += np.dot(sample.transpose(), sample)
        matrixsum /= dim1
        # calculate the covariance
        covariances[i] = matrixsum + diagonal
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    dim1, dim2 = digits.shape
    result = np.zeros((dim1, 10))
    for i in range(dim1):
        const = (-32) * np.log(2 * math.pi)
        dig = digits[i]
        for j in range(10):
            digitmean = means[j]
            digitvar = covariances[j]
            negative = 0.5 * np.log(lg.det(digitvar))
            substract = (dig - digitmean).reshape((1, 64))
            negative += 0.5 * np.dot(np.dot(substract, lg.inv(digitvar)), substract.transpose())
            result[i][j] = const - negative
    return result


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    dim1, dim2 = digits.shape
    result = np.zeros((dim1, 10))
    outcome = generative_likelihood(digits, means, covariances)
    dim1, dim2 = outcome.shape
    for i in range(dim1):
        totalsum = sum(np.exp(outcome[i][l]) for l in range(10))
        avg = totalsum * 0.1
        # calculate p(x|mu, sigma)
        for j in range(dim2):
            result[i][j] = outcome[i][j] + np.log(0.1) - np.log(avg)
    return result


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    dim = labels.shape[0]
    a = [cond_likelihood[i][labels[i].astype(int)] for i in range(dim)]
    total_sum = sum(a)
    avg = total_sum * 1.0 / dim
    return avg


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    filter = np.argmax(cond_likelihood, axis=1)
    return filter


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    # a) part
    acc_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    acc_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(acc_train)
    print(acc_test)

    train_classifier = classify_data(train_data, means, covariances)
    test_classifier = classify_data(test_data, means, covariances)
    # use classifier to compute accuracy,
    train_jude = train_classifier == train_labels
    print("Accuracy on train data is: " + str(train_jude.mean()))
    test_jude = test_classifier == test_labels
    print("Accuracy on test data is: " + str(test_jude.mean()))

    # plot
    for i in range(10):
        cov_matrix = covariances[i]
        evalue = lg.eig(cov_matrix)[0]
        evector = lg.eig(cov_matrix)[1]
        # find the max and corresponding evector
        plt.subplot(2, 5, i + 1)
        # creat subplot
        plt.imshow(evector[:, np.argmax(evalue)].reshape(8,8))
    plt.show()


if __name__ == '__main__':
    main()
