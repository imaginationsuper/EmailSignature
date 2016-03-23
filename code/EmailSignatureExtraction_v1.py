#!/usr/bin/env python
import sys
import re
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
SIZE_DIFF_TOL = 0.001 # tolerance of majority and minority data size difference

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
                    data_neg_vec = SMOTE(data_neg_vec, size_pos, num_neighbors=3)
                elif (size_neg > size_pos*(1+SIZE_DIFF_TOL)):
                    data_pos_vec = SMOTE(data_pos_vec, size_neg, num_neighbors=3)
            models = {"SVC": sklearn.svm.SVC(), \
                      "Logit": sklearn.linear_model.LogisticRegression(), \
                      "DT": sklearn.tree.DecisionTreeClassifier(), \
                      "NBayes": sklearn.naive_bayes.GaussianNB(), \
                      "NNeighbors": sklearn.neighbors.nearest_centroid.NearestCentroid()}
            model_chosen = "NNeighbors"
            accuracys, precisions, recalls, Fscores = cross_validationS(\
                data_pos_vec, data_neg_vec, models[model_chosen], num_cross=NUM_OF_CROSSFOLD) # cross validation
            sFscores.extend(Fscores)
            sRecalls.extend(recalls)
            sPrecisions.extend(precisions)
        stats_Fscore.append(sFscores)
        stats_recall.append(sRecalls)
        stats_precision.append(sPrecisions)
    plt.figure()
    colors = ["red", "blue"]
    modes = ["no-SMOTE", "SMOTE"]
    for i in range(len(stats_Fscore)): # plot statistical summary
        plt.plot(stats_Fscore[i], marker='o', color=colors[i], label=modes[i]+"_Fscore")
        #plt.plot(stats_precision[i], marker='+', color=colors[i], label=modes[i]+"_precision")
        #plt.plot(stats_recall[i], marker='*', color=colors[i], label=modes[i]+"_recall")
    plt.ylim([0, 1.0])
    plt.legend(loc=4, borderaxespad=0.5)
    plt.ylabel("Scores")
    plt.xlabel("Data Sequence")
    plt.savefig("../results/"+model_chosen+"-ValidationStats.png")
    savefile_name = "../results/" + model_chosen + "-ValidationStats.txt"
    fp = open(savefile_name, 'w')
    print "******** Evaluation **********\n"
    fp.write("******** Evaluation **********\n")
    for test_mode in range(2): # print statistical evaluations
        stats_precision[test_mode].sort()
        stats_recall[test_mode].sort()
        stats_Fscore[test_mode].sort()
        p_median = stats_precision[test_mode][len(stats_precision)/2]
        r_median = stats_recall[test_mode][len(stats_recall)/2]
        f_median = stats_Fscore[test_mode][len(stats_Fscore)/2]
        iqr_p = stats_precision[test_mode][int(len(stats_precision)*0.75)] - stats_precision[test_mode][int(len(stats_precision)*0.25)]
        iqr_r = stats_recall[test_mode][int(len(stats_recall)*0.75)] - stats_recall[test_mode][int(len(stats_recall)*0.25)]
        iqr_f = stats_Fscore[test_mode][int(len(stats_Fscore)*0.75)] - stats_Fscore[test_mode][int(len(stats_Fscore)*0.25)]
        print modes[test_mode]
        fp.write(modes[test_mode]+'\n')
        print "\t p_median \t r_median \t f_median"
        fp.write("\t p_median \t r_median \t f_median \n")
        print "\t%.5f \t%.5f \t%.5f" % (p_median, r_median, f_median)
        fp.write("\t%.5f \t%.5f \t%.5f \n" % (p_median, r_median, f_median))
        print "\t iqr_p \t iqr_r \t iqr_f"
        fp.write("\t iqr_p \t iqr_r \t iqr_f \n")
        print "\t%.5f \t%.5f \t%.5f" % (iqr_p, iqr_r, iqr_f)
        fp.write("\t%.5f \t%.5f \t%.5f \n" % (iqr_p, iqr_r, iqr_f))
        print '\n'

def load_data(path_to_data):
    data_pos = []
    data_neg = []
    #sigwords = []
    try:
        fesig = open(path_to_data + "TotalSignaturesA.txt", 'r')
        fecont = open(path_to_data + "train_content.txt", 'r')
        for line in fesig: # read in email signatures
            sigwords = line.lower().split()
            data_pos.append((' '.join(sigwords)).split())
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
        clean_entry = re.sub(r"\n|(\\(.*?){)|}|[@!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]]", ' ', ' '.join(entry))
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
        random.shuffle(sentences)
        model.train(sentences)
    data_pos_vec, data_neg_vec = [], []
    for index in range(len(labeled_data_pos)):
        doc_vec = model.docvecs["DATA_POS_%s"%index]
        data_pos_vec.append(doc_vec.tolist())
    for index in range(len(labeled_data_neg)):
        doc_vec = model.docvecs["DATA_NEG_%s"%index]
        data_neg_vec.append(doc_vec.tolist())
    return data_pos_vec, data_neg_vec

def cross_validation(dataset_pos, dataset_neg, model, num_cross):
    target = [1] * len(dataset_pos) + [0] * len(dataset_neg)
    data_set = dataset_pos + dataset_neg
    accuracys = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="accuracy", cv=num_cross)
    precisions = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="precision", cv=num_cross)
    recalls = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="recall_weighted", cv=num_cross)
    Fscores = sklearn.cross_validation.cross_val_score(model, data_set, target, scoring="f1_weighted", cv=num_cross)
    return accuracys.tolist(), precisions.tolist(), recalls.tolist(), Fscores.tolist()

def cross_validationS(dataset_pos, dataset_neg, model, num_cross):
    data_size = len(dataset_pos) + len(dataset_neg)
    unit_size = int(data_size*1.0 / num_cross)
    dataset_pos_vec = [entry+["pos"] for entry in dataset_pos]
    dataset_neg_vec = [entry+["neg"] for entry in dataset_neg]
    total_dataset = dataset_pos_vec + dataset_neg_vec
    random.shuffle(total_dataset)
    crossfold_dataset = []
    crossfold_target = []
    start = 0
    for crossfold in range(num_cross-1): # divide total dataset into smaller datasets
        unit_dataset = total_dataset[start:start+unit_size]
        unit_target = [entry[-1] for entry in unit_dataset]
        unit_dataset = [entry[:len(entry)-1] for entry in unit_dataset]
        crossfold_dataset.append(unit_dataset)
        crossfold_target.append(unit_target)
        start += unit_size
    unit_dataset = total_dataset[start:]
    unit_target = [entry[-1] for entry in unit_dataset]
    unit_dataset = [entry[:len(entry)-1] for entry in unit_dataset]
    crossfold_dataset.append(unit_dataset)
    crossfold_target.append(unit_target)
    accuracy_list = list()
    Fscores_list = list()
    recall_list = list()
    precision_list = list()
    for crossfold in range(num_cross): # cross-fold validation
        train_dataset = []
        train_target = []
        valid_dataset = []
        valid_target = []
        for i in range(num_cross): # divide into training and validation set
            if i == crossfold:
                valid_dataset.extend(crossfold_dataset[i])
                valid_target.extend(crossfold_target[i])
            else:
                train_dataset.extend(crossfold_dataset[i])
                train_target.extend(crossfold_target[i])
        model_fit = build_model(train_dataset, train_target, model) # training
        valid_predicted = model_fit.predict(valid_dataset).tolist()
        accuracy, precision, recall, Fscore = evaluate_model(valid_target, valid_predicted) # validation
        Fscores_list.append(Fscore)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
    return accuracy_list, precision_list, recall_list, Fscores_list

def build_model(train_data, train_target, model):
    Y = train_target
    X = train_data
    model.fit(X, Y)
    return model

def evaluate_model(target_actual, target_predicted, print_confusion=False):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(target_actual)):
        if target_actual[i] == target_predicted[i] and target_actual[i] == "pos":
            tp += 1
        elif target_actual[i] != target_predicted[i] and target_actual[i] == "pos":
            fn += 1
        elif target_actual[i] == target_predicted[i] and target_actual[i] == "neg":
            tn += 1
        else:
            fp += 1
    accuracy = float(tp+tn) / float(tp+tn+fp+fn+1)
    precision = float(tp) / float(tp+fp+1)
    recall = float(tp) / float(tp+fn+1)
    Fscore = (2*recall*precision) / (recall + precision + 0.00001)
    if print_confusion:
        print str(model)
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
        print "accuracy: %f" % (accuracy)
        print "precision: %f" % (precision)
        print "recall: %f" % (recall)
        print "Fscore: %f" % (Fscore)
        print '\n'
    return accuracy, precision, recall, Fscore

if __name__ == "__main__":
    main()