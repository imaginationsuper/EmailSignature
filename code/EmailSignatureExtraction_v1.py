#!/usr/bin/env python
import sys
import re
import numpy as np
import collections
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import matplotlib.pyplot as plt
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import sklearn.cross_validation

EXIT_FAILURE = -1
EXIT_SUCCESS = 0
NUM_OF_ITERATION = 10 # number of iteration for random order of data entries
NUM_OF_CROSSFOLD = 5 # number of cross-validation
SIZE_DIFF_TOL = 0.1 # tolerance of majority and minority data size difference

def main():
    stats_Fscore, stats_recall, stats_precision  = list(), list(), list()
    data_pos, data_neg = load_data("../data/")
    data_pos, data_neg = data_filter(data_pos), data_filter(data_neg)
    #data_size = max(len(data_pos), len(data_neg))
    for test_mode in range(2):
        sFscores, sRecalls, sPrecisions = list(), list(), list()
        for iteration in range(NUM_OF_ITERATION): # start iteration
            random.seed(iteration)
            random.shuffle(data_pos)
            random.shuffle(data_neg)
            data_pos_vec, data_neg_vec = feature_extraction_Doc2Vec(data_pos, data_neg) # convert to Word Vectors
            size_pos, size_neg = len(data_pos_vec), len(data_neg_vec)
            if test_mode == 1: # apply SMOTE
                if (size_pos > size_neg*(1+SIZE_DIFF_TOL)): # check size and apply SMOTE
                    data_neg_vec = SMOTE(data_neg_vec, size_pos, num_neighbor=3)
                elif (size_neg > size_pos*(1+SIZE_DIFF_TOL)):
                    data_pos_vec = SMOTE(data_pos_vec, size_neg, num_neighbor=3)
            models = {"SVC": sklearn.svm.SVC(), \
                      "Logit": sklearn.linear_model.LogisticRegression(), \
                      "DT": sklearn.tree.DecisionTreeClassifier(), \
                      "NBayes": sklearn.naive_bayes.GaussianNB(), \
                      "NN": sklearn.neighbors.nearest_centroid.NearestCentroid()}
            accuracys, precisions, recalls, Fscores = cross_validation(\
                data_pos_vec, data_neg_vec, models["Logit"], num_cross=NUM_OF_CROSSFOLD) # cross validation
            sFscores.extend(Fscores)
            sRecalls.extend(recalls)
            sPrecisions.extend(precisions)
        stats_Fscore.append(sFscores)
        stats_recall.append(sRecalls)
        stats_precision.append(sPrecisions)
    plt.figure()
    for i in range(len(stats_Fscore)):
        plt.plot(stats_Fscore[i], marker='o')
        plt.plot(stats_precision[i], marker='+')
        plt.plot(stats_recall[i], marker='*')
    plt.show()


def load_data(path_to_data):
    data_pos = []
    data_neg = []
    #sigwords = []
    try:
        fesig = open(path_to_data + "EnronSignatures.txt", 'r')
        fecont = open(path_to_data + "train_content.txt", 'r')
        for line in fesig: # read in email signatures
            sigwords = line.lower().split()
            data_pos.append((' ', join(sigwords)).split())
        for line in fecont: # read in email content
            words = [w.lower() for w in line.strip().split()]
            if len(words) < 2: # filter short words
                continue
            data_neg.append(words)
        fesig.close()
        fecont.close()
    except IOError:
        print "File not exists!"
        sys.exit(EXIT_FAILURE)
    return data_pos, data_neg

def data_filter(raw_text_data):
    clean_text_data = []
    for entry in raw_text_data:
        clean.entry = re.sub(r"\n|(\\(.*?){)|}|[@!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]]", ' ', ' '.join(entry))
        clean_entry = re.sub('\s+', ' ', clean_entry)
        clean_text_data.append(clean_entry.split())
    return clean_text_data

def SMOTE(data_set, total_size, num_neighbors):
    data_set_new = [] # synthesized data
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbors, algorithm="brute").fit(data_set)
    distances, indices = nbrs.kneighbors(data_set)
    origin_size = len(data_set)
    while (origin_size+len(data_set_new) < total_size):
        indexA = random.randint(0, origin_size-1) # pick original data randomly
        data_vec_A = data_set[indexA]
        indexB = indices[indexA][random.randint(0, len(indices[indexA])-1)]
        data_vec_B = data_set[indexB]
        alpha = [random.random() for i in range(len(data_set[0]))] # blending ratio
        new_vec = [alpha[i]*data_vec_A[i] + (1-alpha[i])*data_vec_B[i] for i in range(len(data_vec_A))]
        data_set_new.append(new_vec)
    data_set.extend(data_set_new)
    return data_set

def feature_extraction_Doc2Vec(data_pos, data_neg): # use the word2vec under the hood
    labeled_data_pos = []
    for index, words in enumerate(data_pos):
        sentence = LabeledSentence(words, ["DATA_POS_%s"%index])
        labeled_data_pos.append(sentence)
    labeled_data_neg = []
    for index, words in enumerate(data_neg):
        sentence = LabeledSentence(words, ["DATA_NEG_%s"%index])
        labeled_data_neg.append(sentence)
    model = Doc2Vec(min_count=1, window=20, size=4000, sample=1e-4, negative=5, workers=4)
    sentences = labeled_data_pos + labeled_data_neg
    model.build_vocab(sentences)
    for i in range(5):
        #print "Training iteration %d" %(i)
        random.shuffle(sentences)
        model.train(sentences)
    data_pos_vec, data_neg_vec = [], []
    for index in range(len(labeled_data_pos)):
        doc_vec = model.docvecs["DATA_POS_%s"%index]
        data_pos_vec.append(doc_vec)
    for index in range(len(labeled_data_neg)):
        doc_vec = model.docvecs["DATA_NEG_%s"%index]
        data_neg_vec.append(doc_vec)
    return data_pos_vec, data_neg_vec

def cross_validation(dataset_pos, dataset_neg, model, num_cross):
    #data_size = max(len(data_pos), len(data_neg))
    target = ["pos"] * len(dataset_pos) + ["neg"] * len(dataset_neg)
    data_set = dataset_pos + dataset_neg
    accuracys = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="accuracy_weighted", cv=num_cross)
    precisions = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="precision_weighted", cv=num_cross)
    recalls = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="recall_weighted", cv=num_cross)
    Fscores = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="precision_weighted", cv=num_cross)
    return accuracys.tolist(), precisions.tolist(), recalls.tolist(), Fscores.tolist()


if __name__ == "__main__":
    main()