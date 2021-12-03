# model selected from my csc311 homework
import numpy as np
from typing import Any, List, Dict
from collections import Counter
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
vector = CountVectorizer()

training_data_origin = []


def load_data():
    """Load and preprocess the data using a CountVectorizer
    Return the tuple of 6 components in order x_train, x_test, x_vaild,
    y_train, y_test, y_vaild"""
    remark = []
    file_fake = open("clean_fake.txt", "r")
    arr_fake = file_fake.readlines()
    file_real = open("clean_real.txt", "r")
    arr_real = file_real.readlines()
    for item in arr_fake:
        remark.append("false_news")
    for item in arr_real:
        remark.append("real_news")
        arr_fake.append(item)
    total_news = arr_fake
    # use fit_transform to fit the text file, make it into an array, in order
    # to make it countable about the number of elements
    global vector
    total_news = vector.fit_transform(total_news).toarray()
    mapping = vector.vocabulary_
    # make it a global vector to use
    # the next is spilt
    total_train, test_1, remark_train, test_2 = \
        train_test_split(total_news, remark, random_state=1, train_size=0.7)
    total_test, total_valid, remark_test, remark_valid = \
        train_test_split(test_1, test_2, random_state=1, train_size=0.5)
    # print(total_train)
    return (total_train, total_test, total_valid, remark_train, remark_test,
            remark_valid, mapping)


def model_rater(depth: int, tp: str):
    """Return the accuracy of prediction of the built model based on input data,
     depth"""
    clf_new = DecisionTreeClassifier(criterion=tp, max_depth=depth)
    clf_new.fit(load_data()[0], load_data()[3])
    return accuracy_score(clf_new.predict(load_data()[2]), load_data()[5])


def select_model():
    """Select model and test, return that of the maximum accuracy rate"""
    score_lst = []
    lst_type = ["gini", "entropy"]
    # because of tree structure, we have to make depth large, first consider the
    # min value of the depth.
    print(int((math.log(len(load_data()[0]), 2))))
    # from the output, we know max_depth >= 12 model_rater(dep, type_test),
    # just pick max_depth above 15, 3 relatively small and 2 relatively large

    depth = [15, 20, 25, 50, 100]
    # next is to create prediction, first start by create model based on depth
    # and classifier
    max_rate = 0
    max_dep = 0
    type_best = ""
    for dep in depth:
        for type_test in lst_type:
            print("Model of type " + type_test + " with max_depth " +
                  str(dep) + " has accuracy rate: " +
                  str(model_rater(dep, type_test)))
            if model_rater(dep, type_test) > max_rate:
                max_dep, type_best = dep, type_test
                max_rate = model_rater(dep, type_test)
    print("The best model is " + type_best + " with max_depth " +
          str(max_dep) + " has accuracy rate: " +
          str(model_rater(max_dep, type_best)))
    return max_dep, type_best


def extraction_and_visualization():
    """Return the first two layers of the decision tree"""
    depth, tp = select_model()
    clf = DecisionTreeClassifier(criterion=tp, max_depth=depth)

    clf = clf.fit(load_data()[0], load_data()[3])

    clf = clf.fit(load_data()[0], load_data()[3])

    tree.plot_tree(clf, feature_names=vector.get_feature_names_out(),
                   max_depth=3)
    plt.show()
    return vector.get_feature_names()


def entropy_calculator(prob: float):
    return - prob * math.log(prob, 2) - (1.0 - prob) * math.log(1.0 - prob, 2)


def compute_information_gain():
    # derive key word from the last part
    pos_lst = []
    name_lst = ['donald', 'hillary', 'trumps']
    for item in name_lst:
        pos_lst.append(load_data()[-1][item] - 1)
    count = 0
    for item in load_data()[3]:
        if item == 'real_news':
            count += 1
    prob_false = count * 1.0 / len(load_data()[3])
    total_entro = entropy_calculator(prob_false)
    for i in range(1, 3):
        pos = pos_lst[i]
        lst_feature = []
        for item in load_data()[0]:
            lst_feature.append(item[pos])
        prob_true = sum(lst_feature) * 1.0 / len(lst_feature)
        feature_entro = entropy_calculator(prob_true)
        print("IG(Y|x=" + name_lst[i] + ") = " + str(total_entro - feature_entro))
    #     if "false_news" in item:
    #         count += 1
    # if prob_false == 0:
    #     pass


if __name__ == '__main__':

    select_model()
    load_data()
    extraction_and_visualization()
    compute_information_gain()











