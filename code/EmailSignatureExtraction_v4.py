#!/usr/bin/env python
import sys
import numpy as np
import re
import random
random.seed(0)
from gensim.models.doc2vec import Doc2Vec
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
NUM_OF_BINS = 10 # number of dataset bins
SIZE_DIFF_TOL = 0.001 # tolerance of majority and minority data size difference ratio
SAMPLE_SIZE_RATIO = 0.4 # percentage of total dataset used in the current experiment

def main():
    """
    1. Divide total dataset into several data bins by randomly extracting data entries with given ratio.
    2. Run cross-validation for given numbers of iterations in either SMOTE or non-SMOTE mode.
    3. Report and present statistical evaluations for each data bin.
    """
    stats_Fscores_ns, stats_recalls_ns, stats_precisions_ns = list(), list(), list() # ns for non-SMOTE
    stats_Fscores_ws, stats_recalls_ws, stats_precisions_ws = list(), list(), list() # ws for with SMOTE
    data_pos, data_neg = load_data("../data/")
    data_pos, data_neg = data_filter(data_pos), data_filter(data_neg)
    print "Loading Doc2Vec model ..."
    model_doc2vec = Doc2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) # load Doc2Vec model
    print "Doc2Vec model loading done!"
    models = {"SVC": sklearn.svm.SVC(), \
              "Logit": sklearn.linear_model.LogisticRegression(), \
              "DT": sklearn.tree.DecisionTreeClassifier(), \
              "NBayes": sklearn.naive_bayes.GaussianNB(), \
              "NNeighbors": sklearn.neighbors.nearest_centroid.NearestCentroid()}
    model_chosen = "SVC"
    print "Classifier Type:", model_chosen
    for binIndex in range(NUM_OF_BINS):
        print "Experiment on DataSet#", str(binIndex)
        random.shuffle(data_pos)
        random.shuffle(data_neg)
        size_pos_bin, size_neg_bin = int(len(data_pos)*SAMPLE_SIZE_RATIO), int(len(data_neg)*SAMPLE_SIZE_RATIO)
        data_pos_bin, data_neg_bin = data_pos[:size_pos_bin], data_neg[:size_neg_bin] # dataset bin
        sFscores_iter_ns, sRecalls_iter_ns, sPrecisions_iter_ns = list(), list(), list()
        sFscores_iter_ws, sRecalls_iter_ws, sPrecisions_iter_ws = list(), list(), list()
        for iteration in range(NUM_OF_ITERATION):
            random.seed(iteration)
            random.shuffle(data_pos_bin)
            random.shuffle(data_neg_bin)
            data_pos_vec, data_neg_vec = feature_extraction_Doc2Vec(data_pos_bin, data_neg_bin, model_doc2vec) # convert to doc vectors
            print "non-SMOTE experiment"
            accuracys, precisions, recalls, Fscores = cross_validationS( \
                data_pos_vec, data_neg_vec, models[model_chosen], num_cross=NUM_OF_CROSSFOLD,
                smote_flag=False)  # cross validation
            sFscores_iter_ns.extend(Fscores)
            sRecalls_iter_ns.extend(recalls)
            sPrecisions_iter_ns.extend(precisions)
            print "with SMOTE experiemnt"
            accuracys, precisions, recalls, Fscores = cross_validationS( \
                data_pos_vec, data_neg_vec, models[model_chosen], num_cross=NUM_OF_CROSSFOLD,
                smote_flag=True)  # cross validation
            sFscores_iter_ws.extend(Fscores)
            sRecalls_iter_ws.extend(recalls)
            sPrecisions_iter_ws.extend(precisions)
        stats_Fscores_ns.append(sFscores_iter_ns)
        stats_precisions_ns.append(sPrecisions_iter_ns)
        stats_recalls_ns.append(sRecalls_iter_ns)
        stats_Fscores_ws.append(sFscores_iter_ws)
        stats_precisions_ws.append(sPrecisions_iter_ws)
        stats_recalls_ws.append(sRecalls_iter_ws)
    print "All Experiments Done!"
    save_stats(stats_Fscores_ns, stats_recalls_ns, stats_precisions_ns, stats_Fscores_ws, stats_recalls_ws,\
               stats_precisions_ws, model_name=model_chosen)
    print "Statistics ready!"

def save_stats(stats_Fscores_ns, stats_recalls_ns, stats_precisions_ns, stats_Fscores_ws, stats_recalls_ws, \
               stats_precisions_ws, model_name):
    """
    This calculates statistic metrics for both non-SMOTE and SMOTE experiments, then save into text file and plot
    Fscore medians for all data bins.
    """
    savefile_name = "../results/" + model_name + "-ValidationStats.csv"
    saveplot_name = "../results/" + model_name + "-ValidationStats.png"
    median_Fscores_ns, median_recalls_ns, median_precisions_ns = list(), list(), list()
    median_Fscores_ws, median_recalls_ws, median_precisions_ws = list(), list(), list()
    iqr_Fscores_ns, iqr_recalls_ns, iqr_precisions_ns = list(), list(), list()
    iqr_Fscores_ws, iqr_recalls_ws, iqr_precisions_ws = list(), list(), list()
    for Fscores in stats_Fscores_ns:
        Fscores.sort()
        medianV = Fscores[len(Fscores)/2]
        iqrV = Fscores[int(len(Fscores)*0.75)] - Fscores[int(len(Fscores)*0.25)]
        median_Fscores_ns.append(medianV)
        iqr_Fscores_ns.append(iqrV)
    for recalls in stats_recalls_ns:
        recalls.sort()
        medianV = recalls[len(recalls)/2]
        iqrV = recalls[int(len(recalls) * 0.75)] - recalls[int(len(recalls) * 0.25)]
        median_recalls_ns.append(medianV)
        iqr_recalls_ns.append(iqrV)
    for precisions in stats_precisions_ns:
        precisions.sort()
        medianV = precisions[len(precisions) / 2]
        iqrV = precisions[int(len(precisions) * 0.75)] - precisions[int(len(precisions) * 0.25)]
        median_precisions_ns.append(medianV)
        iqr_precisions_ns.append(iqrV)
    for Fscores in stats_Fscores_ws:
        Fscores.sort()
        medianV = Fscores[len(Fscores) / 2]
        iqrV = Fscores[int(len(Fscores) * 0.75)] - Fscores[int(len(Fscores) * 0.25)]
        median_Fscores_ws.append(medianV)
        iqr_Fscores_ws.append(iqrV)
    for recalls in stats_recalls_ws:
        recalls.sort()
        medianV = recalls[len(recalls) / 2]
        iqrV = recalls[int(len(recalls) * 0.75)] - recalls[int(len(recalls) * 0.25)]
        median_recalls_ws.append(medianV)
        iqr_recalls_ws.append(iqrV)
    for precisions in stats_precisions_ws:
        precisions.sort()
        medianV = precisions[len(precisions) / 2]
        iqrV = precisions[int(len(precisions) * 0.75)] - precisions[int(len(precisions) * 0.25)]
        median_precisions_ws.append(medianV)
        iqr_precisions_ws.append(iqrV)
    try:
        fp = open(savefile_name, 'w')
        fp.write("******** Evaluation **********\n")
        fp.write("non-SMOTE" + '\n')
        for index in range(len(median_Fscores_ns)):
            fp.write("\t Bin#"+str(index)+',')
        fp.write('\n'+"Fscore_median:" + '\n')
        for value in median_Fscores_ns:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "recall_median:" + '\n')
        for value in median_recalls_ns:
            fp.write("\t %.5f,"% (value))
        fp.write('\n' + "precision_median:" + '\n')
        for value in median_precisions_ns:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "iqr_Fscore:" + '\n')
        for value in iqr_Fscores_ns:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "iqr_recall:" + '\n')
        for value in iqr_recalls_ns:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "iqr_precisions:" + '\n')
        for value in iqr_precisions_ns:
            fp.write("\t %.5f," % (value))
        fp.write('\n')
        fp.write("******** Evaluation **********\n")
        for index in range(len(median_Fscores_ws)):
            fp.write("\t Bin#" + str(index)+',')
        fp.write('\n' + "Fscore_median:" + '\n')
        for value in median_Fscores_ws:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "recall_median:" + '\n')
        for value in median_recalls_ws:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "precision_median:" + '\n')
        for value in median_precisions_ws:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "iqr_Fscore:" + '\n')
        for value in iqr_Fscores_ws:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "iqr_recall:" + '\n')
        for value in iqr_recalls_ws:
            fp.write("\t %.5f," % (value))
        fp.write('\n' + "iqr_precisions:" + '\n')
        for value in iqr_precisions_ws:
            fp.write("\t %.5f," % (value))
        fp.close()
        plt.figure()
        numOfBins = len(median_Fscores_ns)
        dataBinIndex = [i for i in range(numOfBins)]
        plt.plot(dataBinIndex, median_Fscores_ns, color="blue", linewidth=2, label="non-SMOTE_median")
        plt.plot(dataBinIndex, median_Fscores_ws, color="green", linewidth=2, label="SMOTE_median")
        plt.plot(dataBinIndex, iqr_Fscores_ns, color="blue", linestyle='--', linewidth=2, label="non-SMOTE_iqr")
        plt.plot(dataBinIndex, iqr_Fscores_ws, color="green", linestyle='--', linewidth=2, label="SMOTE_iqr")
        plt.ylim([0, 1.0])
        plt.legend(loc=0, borderaxespad=0.5, frameon=False)
        plt.ylabel("F-scores")
        plt.xlabel("Data Bin Index")
        plt.savefig(saveplot_name)
    except IOError:
        print "Cannot open save files!"

def load_data(path_to_data):
    """
    This will load positive(email signature components) and negative(email body) data from the given directory,
     and return them as two word lists.
    """
    data_pos = []
    data_neg = []
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
    """
    This will filter the raw text lists by regular expressions, and only keep letter character tokens.
    """
    clean_text_data = []
    for entry in raw_text_data:
        clean_entry = re.sub(r"\n|(\\(.*?){)|}|[@!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]]", ' ', ' '.join(entry))
        clean_entry = re.sub('\s+', ' ', clean_entry)
        clean_text_data.append(clean_entry.split())
    return clean_text_data

def SMOTE(data_set, total_size, num_neighbors):
    """
    Given the minority dataset, it will find the nearest neighbor data points for each original data and create
     new data points between the nearest neighbor points randomly. This process will stop when the total dataset
     size reach the expected total_size. And synthetic dataset will be blended with the original dataset.
    """
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
    random.shuffle(data_set)
    return data_set

def feature_extraction_Doc2Vec(data_pos, data_neg, model_doc2vec): # use the word2vec under the hood
    """
    Given the trained Doc2Vec model, it will convert text dataset(word lists) into numerical vectors.
    The conversion is done word by word, and document vector is the average of all word vectors in the current
      document. Un-indexed word will be skipped.
    """
    vec_size = 300
    data_pos_vec, data_neg_vec = [], []
    for term in data_pos:
        raw_vecs = np.zeros(vec_size)
        vec_num = 0
        for item in term:
            try:
                raw_vecs = np.add(raw_vecs, model_doc2vec[item])
                vec_num += 1
            except:
                pass
        doc_vec = raw_vecs / float(vec_num+0.00001)
        data_pos_vec.append(doc_vec.tolist())
    for term in data_neg:
        raw_vecs = np.zeros(vec_size)
        vec_num = 0
        for item in term:
            try:
                raw_vecs = np.add(raw_vecs, model_doc2vec[item])
                vec_num += 1
            except:
                pass
        doc_vec = raw_vecs / float(vec_num + 0.00001)
        data_neg_vec.append(doc_vec.tolist())
    return data_pos_vec, data_neg_vec

def cross_validationS(dataset_pos_veclist, dataset_neg_veclist, model, num_cross, smote_flag=False):
    """
    Apply cross-validation to the dataset(list of doc vectors). SMOTE method is optional. model is used for classifier trainig and
    validation.
    """
    dataset_pos, dataset_neg = list(dataset_pos_veclist), list(dataset_neg_veclist)
    data_size = len(dataset_pos) + len(dataset_neg)
    unit_size = int(data_size*1.0 / num_cross)
    dataset_pos_vec = [entry+["pos"] for entry in dataset_pos]
    dataset_neg_vec = [entry+["neg"] for entry in dataset_neg]
    total_dataset = dataset_pos_vec + dataset_neg_vec
    random.shuffle(total_dataset)
    crossfold_dataset = []
    crossfold_target = []
    start = 0
    for crossfold in range(num_cross-1): # divide total dataset evenly into smaller datasets
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
        if smote_flag:
            train_pos_dataset = [entry for index, entry in enumerate(train_dataset) if train_target[index]=="pos"]
            train_neg_dataset = [entry for index, entry in enumerate(train_dataset) if train_target[index]=="neg"]
            size_pos = len(train_pos_dataset)
            size_neg = len(train_neg_dataset)
            if (size_pos > size_neg*(1+SIZE_DIFF_TOL)): # check size and apply SMOTE
                train_neg_dataset = SMOTE(train_neg_dataset, size_pos, num_neighbors=4)
            elif (size_neg > size_pos*(1+SIZE_DIFF_TOL)):
                train_pos_dataset = SMOTE(train_pos_dataset, size_neg, num_neighbors=4)
            size_pos = len(train_pos_dataset)
            size_neg = len(train_neg_dataset)
            #print size_pos, size_neg
            train_dataset = train_pos_dataset + train_neg_dataset
            train_target = size_pos*["pos"] + size_neg*["neg"]
        model_fit = build_model(train_dataset, train_target, model) # training
        valid_predicted = model_fit.predict(valid_dataset).tolist()
        accuracy, precision, recall, Fscore = evaluate_model(valid_target, valid_predicted) # validation
        Fscores_list.append(Fscore)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
    return accuracy_list, precision_list, recall_list, Fscores_list

def build_model(train_data, train_target, model):
    """
    Training classifier by given dataset and labels.
    """
    Y = train_target
    X = train_data
    model.fit(X, Y)
    return model

def evaluate_model(target_actual, target_predicted, print_confusion=False):
    """
    Evaluate the classifier using statistical metrics. Actual and predicted results should be provided.
    """
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