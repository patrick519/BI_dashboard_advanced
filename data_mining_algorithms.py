import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import itertools
import seaborn as sns
from collections import defaultdict
from sklearn import tree, svm, datasets, linear_model
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Source:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Declaring global vars
df = None
X = None
Y = None
x_train = None
x_validation = None
x_test = None
y_train = None
y_validation = None
y_test = None

# Read data from hardcoded input file
def load_data(tsize,feature_names) :
    global df, X, Y, x_train, x_validation, x_test, y_train, y_validation, y_test, f_names
    df          = pd.read_csv(filepath_or_buffer = "customer_data.csv")
    #df          = pd.read_csv(filepath_or_buffer = "customer_dataV2.csv")
    f_names = feature_names
    features = df[feature_names]
    target   = df["repurchase"]

    X = features
    Y = target

    x_tr_val, x_test, y_tr_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_tr_val, y_tr_val, test_size=0.2, random_state=42)
    print ("randomized precision: ", 1-Y.mean())
    return;

def correlation():
    sns.set_style("whitegrid", {'axes.grid' : False})
    corr = df.corr()
    sns.heatmap(corr)
    plt.xticks(rotation=70)
    plt.yticks(rotation=0)
    plt.subplots_adjust(bottom=0.35,left=0.25)
    title='Corellation Matrix'
    plt.savefig(title)

# Accuracy of a confusion matrix
def accuracy_of_cnf(cnf_matrix) :
    return (cnf_matrix[0][0] + cnf_matrix[1][1]) / (cnf_matrix[0][0] + cnf_matrix[0][1] + cnf_matrix[1][0] + cnf_matrix[1][1])*100;

# Classify with standard decision tree
def cnf_decision_tree(depth):
    print("\nStandard decision tree:")
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_of_cnf(cnf_matrix)
    print("Accuracy of decision tree is: " , accuracy)
    title = 'Classify: Standard Decision Tree'
    plt.figure(title)
    plot_confusion_matrix(cnf_matrix, classes=['No Future Sale', 'Future Sale'], normalize=True, title=title)
    #plt.savefig(title)
    return;

# Classify with Random forest
def cnf_random_forest(depth):
    print("\nRandom forest:")
    rf_clf = RandomForestClassifier(n_estimators=10, max_depth = depth)
    rf_clf = rf_clf.fit(x_train, y_train)
    y_pred = rf_clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_of_cnf(cnf_matrix)
    print("Accuracy of Random Forests is: " , accuracy)
    title = 'Classify: Random Forest'
    plt.figure(title)
    plot_confusion_matrix(cnf_matrix, classes=['No Future Sale', 'Future Sale'], normalize=True, title=title)
    #plt.savefig(title)
    return;

# Classify with logistic regression
def cnf_logistic_regression() :
    print("\nLogistic regression:")
    log_clf = LogisticRegression()
    log_clf = log_clf.fit(x_train, y_train)
    y_pred = log_clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_of_cnf(cnf_matrix)
    print("Accuracy of logistic regression is: " , accuracy, "\n")
    title = 'Classify: Logistic Regression Model'
    plt.show(block=False)
    plt.figure(title)
    plot_confusion_matrix(cnf_matrix, classes=['No Future Sale', 'Future Sale'], normalize=True, title=title)
    plt.savefig(title)
    return;

# Classify with support vector machine
def cnf_support_vector_machine() :
    print("\nSupport vector machine:")
    svm_clf = svm.SVC()
    svm_clf = svm_clf.fit(x_test, y_test)  # we interchange the role of train and validation set solely for the purpose of speed. Training a support vector machine is very slow; its complexity rises quadratically with the number of samples
    y_pred = svm_clf.predict(x_train)
    cnf_matrix = confusion_matrix(y_train, y_pred)
    accuracy = accuracy_of_cnf(cnf_matrix)
    print("Accuracy of support vector machine is: " , accuracy, "\n")
    title = 'Classify: Support Vector Machine'
    plt.figure(title)
    plot_confusion_matrix(cnf_matrix, classes=['No Future Sale', 'Future Sale'], normalize=True, title=title)
    #plt.savefig(title)
    return;

# Classify with Multi Layer Perceptron
def cnf_multi_layer_perceptron() :
    print("\nMulti layer perceptron:")
    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10), random_state=1)
    mlp_clf = mlp_clf.fit(x_train, y_train)
    y_pred = mlp_clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_of_cnf(cnf_matrix)
    print("Accuracy of Multi-layer Perceptron is: " , accuracy, "\n")
    title = 'Classify: Multi Layer Perceptron'
    plt.figure(title)
    plot_confusion_matrix(cnf_matrix, classes=['No Future Sale', 'Future Sale'], normalize=True, title=title)
    #plt.savefig(title)
    return;



# Compute performance of 4 classifier algorithms
def cross_val_performance():
    print("Cross validation: ")
    rf_clf = RandomForestClassifier(n_estimators=10, max_depth = 7)
    log_clf = LogisticRegression()
    svm_clf = svm.SVC()
    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10), random_state=1)


    # Random Forest
    print("\nRandom Forest: ")
    crossValScore = cross_val_score(rf_clf, x_train, y_train, scoring='accuracy', cv = 10) # data input must be shuffled. This is done in train_test_split.
    print("\nCross validation scores:")
    print(crossValScore)
    accuracy = crossValScore.mean() * 100
    print("Accuracy of Random Forests is: " , accuracy)


    # Logarithmic Regression
    print("\n\nLog:")
    crossValScore = cross_val_score(log_clf, x_train, y_train, scoring='accuracy', cv = 10)
    print("\nCross validation scores:")
    print(crossValScore)
    accuracy = crossValScore.mean() * 100
    print("Accuracy of Logistic Regression is: " , accuracy)


    # Support Vector Machine    (performs better if you take only few features, especially when all of the features are time-related)
    print("\n\nSVM:")
    crossValScore = cross_val_score(svm_clf, x_test, y_test, scoring='accuracy', cv = 10)
    print("\nCross validation scores:")
    print(crossValScore)
    accuracy = crossValScore.mean() * 100
    print("Accuracy of SVM is: " , accuracy)


    # Multi Layer Perceptron
    print("\n\nMulti Layer Perceptron:")
    crossValScore = cross_val_score(mlp_clf, x_train, y_train, scoring='accuracy', cv = 10)
    print("\nCross validation scores:")
    print(crossValScore)
    accuracy = crossValScore.mean() * 100
    print("Accuracy of Multi Layer Perceptron is: " , accuracy)

    return;



# Random forest: prints accuracy for different maxdepths (plotted in R)
def random_forest_vary_maxdepth(start, end) :
    print("\nRandom forest with varying maxdepth:")
    for i in range(start,end+1) :
        rf_clf = RandomForestClassifier(n_estimators=10, max_depth = i)
        rf_clf = rf_clf.fit(x_train, y_train)
        y_pred = rf_clf.predict(x_test)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_of_cnf(cnf_matrix)
        if(i != 1) :
            print(",",end="")
        print(accuracy,end="")
    print("")
    return;

def random_forest_without_maxdepth() :
    print("\nRandom forest without maxdepth:")
    rf_clf = RandomForestClassifier(n_estimators=10)
    rf_clf = rf_clf.fit(x_train, y_train)
    y_pred = rf_clf.predict(x_validation)
    cnf_matrix = confusion_matrix(y_validation, y_pred)
    accuracy = accuracy_of_cnf(cnf_matrix)
    print(accuracy)

# Logistic regression: prints accuracy for different tolerances
def logistic_regression_vary_tolerance() :
    print("\nLogistic regression with varying tolerance:")
    tol_arr = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.075, 0.1, 0.3, 0.5]
    for i in range(0, len(tol_arr)) :
        log_clf = LogisticRegression(tol=tol_arr[i])
        log_clf = log_clf.fit(x_train, y_train)
        y_pred = log_clf.predict(x_validation)
        cnf_matrix = confusion_matrix(y_validation, y_pred)
        accuracy = accuracy_of_cnf(cnf_matrix)
        if(i != 0) :
            print(",",end="")
        print(accuracy,end="")
    print("")
    return;

# Support vector machine: prints accuracy for different tolerances
def support_vector_machine_vary_penalty() :
    print("\nSupport vector machine with varying penalty of the error term:")
    penalty_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4]
    for i in range(0, len(penalty_arr)) :
        svm_clf = svm.SVC(C=penalty_arr[i])
        svm_clf = svm_clf.fit(x_validation, y_validation)  # we interchange the role of train and validation set solely for the purpose of speed.
        y_pred = svm_clf.predict(x_train)
        cnf_matrix = confusion_matrix(y_train, y_pred)
        accuracy = accuracy_of_cnf(cnf_matrix)
        if(i != 0) :
            print(",",end="")
        print(accuracy,end="")
    print("")
    return;

# Support vector machine: prints accuracy for different number of layers
def multi_layer_perceptron_vary_number_of_layers() :
    print("\nMulti layer perceptron with varying number of layers:")

    layer_arr = [30, [15,15], [10,10,10], [8,8,7,7], [6,6,6,6,6], [5,5,5,5,5,5], [5,5,4,4,4,4,4], [4,4,4,4,4,4,3,3], [4,4,4,3,3,3,3,3,3], [3,3,3,3,3,3,3,3,3,3]]
    for i in range(0, len(layer_arr)) :
        mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layer_arr[i], random_state=1)
        mlp_clf = mlp_clf.fit(x_validation, y_validation)  # we interchange the role of train and validation set solely for the purpose of speed.
        y_pred = mlp_clf.predict(x_train)
        cnf_matrix = confusion_matrix(y_train, y_pred)
        accuracy = accuracy_of_cnf(cnf_matrix)
        if(i != 0) :
            print(",",end="")
        print(accuracy,end="")
    print("")
    return;

# Support vector machine: prints accuracy for different learning rates alpha.   since the runtime of this function is a couple of minutes be sure selecting very few features beforehand
def multi_layer_perceptron_vary_learning_rate() :
    print("\nMulti layer perceptron with varying learning rate:")

    alpha_arr = [i/50 for i in range(1,51)]
    for i in range(0, len(alpha_arr)) :
        mlp_clf = MLPClassifier(solver='lbfgs', alpha=alpha_arr[i], hidden_layer_sizes=(20,10), random_state=1)
        mlp_clf = mlp_clf.fit(x_validation, y_validation)  # we interchange the role of train and validation set solely for the purpose of speed.
        y_pred = mlp_clf.predict(x_train)
        cnf_matrix = confusion_matrix(y_train, y_pred)
        accuracy = accuracy_of_cnf(cnf_matrix)
        if(i != 0) :
            print(",",end="")
        print(accuracy,end="")
    return;

#main
#uncomment 'feature_names' to use the listed features you want
feature_names = ["numberOfTransactions","totalSpending","firstTransactionDate","lastTransactionDate","lifeSpan","avgTimeBetweenTransactions","maxTransactionPrice","avgTransactionPrice","countryCode"]
#feature_names = ["transactionsLastMonth","transactions2ndLastMonth","transactions3rdLastMonth","transactions4thLastMonth","transactionsLastWeek","transactionsLast2Weeks","transactionsLast2Months","transactionsLast4Months","transactionsLast8Months"]
#feature_names = ["spendingLastMonth","spending2ndLastMonth","spending3rdLastMonth","spending4thLastMonth","spendingLastWeek","spendingLast2Weeks","spendingLast2Months","spendingLast4Months","spendingLast8Months"]
#feature_names = ["numberOfTransactions","totalSpending","firstTransactionDate","lastTransactionDate","lifeSpan","avgTimeBetweenTransactions","maxTransactionPrice","avgTransactionPrice","countryCode","transactionsLastMonth","transactions2ndLastMonth","transactions3rdLastMonth","transactions4thLastMonth","transactionsLastWeek","transactionsLast2Weeks","transactionsLast2Months","transactionsLast4Months","transactionsLast8Months"]
#feature_names = ["spendingLastMonth","spending2ndLastMonth","spending3rdLastMonth","spending4thLastMonth","spendingLastWeek","spendingLast2Weeks","spendingLast2Months","spendingLast4Months","spendingLast8Months","transactionsLastMonth","transactions2ndLastMonth","transactions3rdLastMonth","transactions4thLastMonth","transactionsLastWeek","transactionsLast2Weeks","transactionsLast2Months","transactionsLast4Months","transactionsLast8Months"]

load_data(0.25,feature_names)

correlation()

# generation of confusion matrices
cnf_decision_tree(10)
cnf_random_forest(4)
cnf_logistic_regression()
cnf_support_vector_machine()
cnf_multi_layer_perceptron()

# Performance of all classifiers using cross-validation
#cross_val_performance()

# Parameter optimization of different classifiers
#random_forest_vary_maxdepth(1, 10)
#random_forest_without_maxdepth()
#logistic_regression_vary_tolerance()
#support_vector_machine_vary_penalty()
#multi_layer_perceptron_vary_number_of_layers()
#multi_layer_perceptron_vary_learning_rate() # only finishes after one closes all the generated images

plt.show()

