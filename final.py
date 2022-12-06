
import Levenshtein as leven
from Levenshtein import distance as levenDist
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import fuzzywuzzy
from fuzzywuzzy import fuzz, process



import numpy as np

import math

'''
loss="hinge" -> SVM
loss="log_loss" -> logistic regression

penalty="l2", "l1", "elasticnet" (l2 is default)

max_iter, 1000 is default

tol, stopping parameter
early_stopping, true or false
n_iter_no_change,
score, 
'''
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier

import imblearn
from imblearn.ensemble import BalancedBaggingClassifier


def __main__():


    ac = f1 = 0

    '''make new features files, OTHERWISE COMMENT THIS OUT'''
    train = TextFileToFeatureMatrix("train_with_label.txt")
    x_train, y_train = train.makeFeatures()
    
    dev = TextFileToFeatureMatrix("dev_with_label.txt")
    x_dev, y_dev = dev.makeFeatures()

    test = TextFileToFeatureMatrix("test_without_label.txt")
    x_test = test.makeFeatures()
        #scale
    sc = StandardScaler()
    sc.fit(x_train)
    
    x_train_scaled = sc.transform(x_train)
    makeCSV("x_train_scaled.txt", x_train_scaled)
    makeCSV("y_train.txt", y_train)
    
    x_dev_scaled = sc.transform(x_dev)
    makeCSV("x_dev_scaled.txt", x_dev_scaled)
    makeCSV("y_dev.txt", y_dev)
    
    x_test_scaled = sc.transform(x_test)
    makeCSV("x_test_scaled.txt", x_test_scaled)


    '''make features from existing feature files, OTHERWISE COMMENT THIS OUT'''
##    x_test_scaled = makeFeaturesFromCSV("x_test_scaled.txt");
##    x_dev_scaled = makeFeaturesFromCSV("x_dev_scaled.txt");
##    x_train_scaled = makeFeaturesFromCSV("x_train_scaled.txt");
##    y_train = makeFeaturesFromCSV("y_train.txt")
##    y_dev = makeFeaturesFromCSV("y_dev.txt")
    

    '''mlp'''
##    mlp = MLPClassifier( activation = "logistic")
##    mlp.fit(x_train_scaled, np.ravel(y_train))
##        prediction = mlp.predict(x_dev_scaled)
    layers = (1000,200,100)

    '''0.4 works well'''
    bagged_mlp = BalancedBaggingClassifier(base_estimator = MLPClassifier( activation = "logistic", hidden_layer_sizes=layers), n_jobs = 3,sampling_strategy=0.6,replacement=False)
    bagged_mlp.fit(x_train_scaled, np.ravel(y_train))
    prediction_bagged = bagged_mlp.predict(x_dev_scaled)

    unbagged_mlp = MLPClassifier( activation = "relu", hidden_layer_sizes = layers)
    unbagged_mlp.fit(x_train_scaled, np.ravel(y_train))
    prediction_unbagged = unbagged_mlp.predict(x_dev_scaled)
    
    print("predicting on bagging")
    ac = accuracy(y_dev, prediction_bagged)
    f1_1 = f1_score(y_dev, prediction_bagged)
    print("accuracy: ", ac)
    print("f1: ", f1_1)


    print("predicting on unbagging")
    ac = accuracy(y_dev, prediction_unbagged)
    f1_2 = f1_score(y_dev, prediction_unbagged)
    print("accuracy: ", ac)
    print("f1: ", f1_2)

    winner_model = None
    if f1_2 > f1_1:
        print("winner is unbagged")
        winner_model = unbagged_mlp
    else:
        print("winner is bagged")
        winner_model = bagged_mlp
    test_prediction = winner_model.predict(x_test_scaled)
    
    
    '''logistic regression'''
##    
##    while True:
##        svmSGD_scaled = SGDClassifier(loss="hinge", tol = 0.0001, max_iter=10**8).fit(x_train_scaled, y_train)
##        prediction = svmSGD_scaled.predict(x_dev_scaled)
##        ac = accuracy(y_dev, prediction)
##        f1 = f1_score(y_dev, prediction)
##        if ac  > 0.69:
##            break
##    y_prediction = svmSGD_scaled.predict(x_test_scaled)
##

    f = open("SeanBritt_test_result.txt", "w")

    for i in range(0, len(test_prediction)):
        string = "test_id_" + str(i) + "\t" + str(test_prediction[i]) +"\n"
        f.write(string)

    f.close()
    
##    f=open("SeanBritt_test_result.txt", "r")
##    f.readlines()
##
##    f.close()


    exit

def accuracy(gold, pred):
    count = 0
    for i in range(0, len(pred)):
        if gold[i] == pred[i]:
            count += 1
        

    return (count/len(gold))


def makeCSV(file, features=None):
    f = open(file, "w")

    if len(features) == 0:
        print("no features to export")

    else:
        if isinstance(features, np.ndarray):
            for row in features:
                line = ''
                for num in row:
                    line+=str(num) + ", "
                line = line[0:-2]
                line += '\n'                
                f.write(line)
        else:
            for num in features:
                line = ''
                line += str(num)
                line += '\n'
                f.write(line)
    
    f.close()

        
def makeFeaturesFromCSV(file):
        f = open(file, "r")
        features = []
        for line in f:
            line = line.strip('\n')
            x_row = line.split(',')
            x_row = [float(i) for i in x_row]
            features.append(x_row)
        return features
            
            

class TextFileToFeatureMatrix:
    
    def __init__ (self, file):
        self.file = file
        self.vectorCount = 0
        self.trainDimensions = 7
        self.dataID = []
        self.data = []
        self.label = []
        self.features = 0
        self.stemmer = PorterStemmer()
        self.rawDataLines = []
        self.stop_words = set(stopwords.words('english'))

         
   
    def makeFeatures(self):

        #open the text file for reading
        f = open(self.file, "r")
        
        '''
        Look at each line of the file and turn it into a sentence pair and a label
        '''
        #initial splitting of the lines by /t and sending the info where it needs to go
        for line in f:
            
            tabSplit = line.split("\t")

            #these are the two sentences and the label for the pair
            '''
            1. make the sentence lower case
            2. tokenize the sentence
            3. clean the token string 
            '''

            sent1 = self.cleanTok(nltk.word_tokenize(tabSplit[1].lower()))
            sent2 = self.cleanTok(nltk.word_tokenize(tabSplit[2].lower()))
            self.data.append((sent1, sent2))

            #this checks for the test case
            if len(tabSplit) > 3:
                self.label.append(int(tabSplit[-1]))

            #keep the count of sentence pairs.  this will be equal to the length of self.data and self.label
            self.vectorCount += 1

        #start the feature list with all 0s
        self.features = np.zeros(shape=(self.vectorCount, self.trainDimensions))
        for i in range(0, self.vectorCount):
            sent1 = self.data[i][0]
            sent2 = self.data[i][1]
        
            weights = [(1, 0, 0, 0), (1./2., 1./2., 0, 0), (1./3., 1./3., 1./3., 0), (1./4., 1./4., 1./4., 1./4.)]
            bleu_scores1 = sentence_bleu([sent1], sent2, weights, smoothing_function=SmoothingFunction().method1)
            
            self.features[i][0] = bleu_scores1[0]
            self.features[i][1] = bleu_scores1[1]
            self.features[i][2] = bleu_scores1[2]
            self.features[i][3] = bleu_scores1[3]

            self.features[i][4] = math.exp(abs(len(sent1)-len(sent2)))

            self.features[i][5] = levenDist(sent1, sent2)
            self.features[i][6] = fuzz.token_sort_ratio(sent1, sent2)
            

        f.close()
        if len(self.label) == 0:
            return self.features
        else:        
            return self.features, self.label


    '''
    if a token is longer than 1 character and is not included in the stop_words
    1. stem the token
    2. add it to the list of tokens to return

    then, return the list of cleaned tokens
    '''

    def cleanTok(self, line):
        cleantoks = []
        
        for tok in line:
            if(len(tok) > 1 and (tok not in self.stop_words)):
                cleantoks.append(self.stemmer.stem(tok))
        return cleantoks
    
    '''
    -----------------FEATURE CONGLOMERATION
    '''

    #compare the lengths of the tokens as a feature 
    def ft_length(self, line1, line2):
        return 1/(max(1, (len(line1)-len(line2))**2) )
        

    
__main__()
