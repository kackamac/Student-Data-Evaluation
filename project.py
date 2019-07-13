import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# =============================================================================

# PREPROCESSING DATASET

# replace nonnumeracil values by numerical ones
# param: dataset = daat to process
def preprocessDataset(dataset):    
    # replace string values by numerical
    school = {'GP': 0,'MS': 1} 
    dataset.school = [school[item] for item in dataset.school] 
    
    sex = {'M': 0,'F': 1} 
    dataset.sex = [sex[item] for item in dataset.sex] 
    
    address = {'U': 0,'R': 1} 
    dataset.address = [address[item] for item in dataset.address] 
    
    famsize = {'GT3': 0,'LE3': 1} 
    dataset.famsize = [famsize[item] for item in dataset.famsize] 
    
    Pstatus = {'A': 0,'T': 1} 
    dataset.Pstatus = [Pstatus[item] for item in dataset.Pstatus] 
    
    Mjob = {'at_home': 0,'health': 1,'teacher': 2,'other': 3,'services': 4 } 
    dataset.Mjob = [Mjob[item] for item in dataset.Mjob] 
    
    Fjob =  {'at_home': 0,'health': 1,'teacher': 2,'other': 3,'services': 4 } 
    dataset.Fjob = [Fjob[item] for item in dataset.Fjob] 
    
    reason = {'course': 0,'home': 1,'reputation': 2,'other': 3} 
    dataset.reason = [reason[item] for item in dataset.reason] 
    
    guardian = {'mother': 0,'father': 1, 'other':2} 
    dataset.guardian = [guardian[item] for item in dataset.guardian] 
    
    #replace yes/no by 1/0 in the same manner
    schoolsup = {'no': 0,'yes': 1} 
    dataset.schoolsup = [schoolsup[item] for item in dataset.schoolsup] 
    famsup = {'no': 0,'yes': 1} 
    dataset.famsup = [famsup[item] for item in dataset.famsup] 
    paid = {'no': 0,'yes': 1} 
    dataset.paid = [paid[item] for item in dataset.paid] 
    activities = {'no': 0,'yes': 1} 
    dataset.activities = [activities[item] for item in dataset.activities] 
    nursery = {'no': 0,'yes': 1} 
    dataset.nursery = [nursery[item] for item in dataset.nursery] 
    higher = {'no': 0,'yes': 1} 
    dataset.higher = [higher[item] for item in dataset.higher] 
    internet = {'no': 0,'yes': 1} 
    dataset.internet = [internet[item] for item in dataset.internet] 
    romantic = {'no': 0,'yes': 1} 
    dataset.romantic = [romantic[item] for item in dataset.romantic] 
    return dataset

# replace final grade with 1 passed (grade>=10) or 0 failed (grade<10)
# param: Y = column to replace
def binarizePredictedValues(Y):    
    for i in range(0,len(Y)):
        if(Y[i]<10):
            Y[i]=0
        else:
            Y[i]=1  
    return Y

# =============================================================================

# FEATURE EXTRACTION

# MODEl 1 RF: important feature extraction with random forests
# params: X data, Y target
def featureExtractionRandomForestClassifier(X,Y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, Y)
    #print importance score for each attribute - the larger score the more important
    #print(model.feature_importances_)
    return model

# MODEL2 DT: important feature extraction with DecisionTreeClassifier
# params: X data, Y target
def featureExtractionDecisionTrees(X,Y):
    model = DecisionTreeClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    #print importance score for each attribute - the larger score the more important
    #print(model.feature_importances_)
    return model

# MODEL3 ET: important feature extraction with ExtraTreesClassifier
# params: X data, Y target
def featureExtractionExtraTreesClassifier(X,Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    #print importance score for each attribute - the larger score the more important
    #print(model.feature_importances_)
    return model
    
# =============================================================================
    
# VIZUALIZATION
    
# create a bar plot
# params: dataset, model, feature names
def visualization(dataset, model, feature_names, title):
    feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(title)
    plt.legend()
    plt.show()

# =============================================================================    

#PRINCIPAL COMPONENT ANALYSIS

# PCA1: feature extraction by Principal Component Analysis - finding k main component in dataset 
# params: k number of components, l last column of dataset
def pca(dataset, k,l):
    array = dataset.values
    X = array[:,0:l]
    Y = array[:,32]
    pca = PCA(n_components=k)
    fit = pca.fit(X)
    X_pca = pca.transform(X)
    # summarize components
    print("Explained Variance: ", fit.explained_variance_ratio_) 
    
    # transformed dataset (k principal components)
    #print(fit.components_) 
    return X_pca


# =============================================================================  
    
# PREDICITON FROM DATA
    
# MODEL1 RF: split data to train and test and try to predict correct grade according to Random Forests
# params: X data, Y target
def predictionRandomForests(X, Y):
    model = RandomForestClassifier(n_estimators=100)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    return metrics.accuracy_score(Y_test, Y_pred)

#MODEL2 DT: split data to train and test and try to predict correct grade according to Decision Trees
# params: X data, Y target
def predictionDecisionTree(X, Y):
    model = DecisionTreeClassifier()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    return metrics.accuracy_score(Y_test, Y_pred)

#MODEL3 ET: split data to train and test and try to predict correct grade according to Extra Trees
# params: X data, Y target
def predictionExtraTrees(X, Y):
    model = ExtraTreesClassifier()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    return metrics.accuracy_score(Y_test, Y_pred)

# =============================================================================  

    
def main():
    # lOAD AND PREPROCESS DATASET
    # 2 datasets student-por.csv, student-mat.csv
    dataset = pd.read_csv('student-por.csv', sep=';')
    dataset = preprocessDataset(dataset)
    
    # now the data is ready to process
    array = dataset.values
    X = array[:,0:32]
    Y = array[:,32]
    
    #FEATURE EXTRACTION
    
    print("FEATURE EXTRACTION")
    #1 with Random Forests 
    model = featureExtractionRandomForestClassifier(X,Y)
    feature_names = dataset.columns.values[:-1] #do not need predicted column for that
    visualization(dataset, model, feature_names, "Important features - Random forest")
    #2 with Decision Trees
    model = featureExtractionDecisionTrees(X,Y)
    visualization(dataset, model, feature_names, "Important features - Decision tree")
    #3 with Extra Trees
    model = featureExtractionExtraTreesClassifier(X,Y)
    visualization(dataset, model, feature_names, "Important features - Extra tree")
    
    
    
    # =============================================================================
    
    #PRINCIPAL COMPONENT ANALYSIS
    
    transformed_X = pca(dataset, 6, 32)
   
     
    # =============================================================================
    # WHOLE DATASET PREDICTIONS    
    print("DATASET PREDICTIONS ALL GRADES")
    # RANDOM FORESTS    
    #1: RandomForest to predict resultant score on whole dataset
    acc1 = predictionRandomForests(array[:,0:32], Y)
    print("Accuracy1 (RF-A, allData):",acc1)    
    #2: now lets do the same but only with 3 main components - G1, G2, abscences
    acc2 = predictionRandomForests(array[:,29:32], Y)
    print("Accuracy2 (RF-A, main3comp):",acc2)    
    #3: now lets do the same but with data transformed from PCA
    acc3 = predictionRandomForests(transformed_X, Y)
    print("Accuracy3 (RF-A, pcaX):",acc3)
    
    # DECISION TREES
    #4: DecisionTree to predict resultant score on whole dataset
    acc4 = predictionDecisionTree(array[:,0:32], Y)
    print("Accuracy4 (DT-A, allData):",acc4)    
    #5: now lets do the same but only with 3 main components - G1, G2, abscences
    acc5 = predictionDecisionTree(array[:,29:32], Y)
    print("Accuracy5 (DT-A, main3comp):",acc5)    
    #6: now lets do the same but with data transformed from PCA
    acc6 = predictionDecisionTree(transformed_X, Y)
    print("Accuracy6 (DT-A, pcaX):",acc6)
    
    # EXTRA TREES
    #7: ExtraTrees to predict resultant score on whole dataset
    acc7 = predictionExtraTrees(array[:,0:32], Y)
    print("Accuracy7 (ET-A, allData):",acc7)   
    #8: now lets do the same but only with 3 main components - G1, G2, abscences
    acc8 = predictionExtraTrees(array[:,29:32], Y)
    print("Accuracy8 (ET-A, main3comp):",acc8)    
    #9: now lets do the same but with data transformed from PCA
    acc9 = predictionExtraTrees(transformed_X, Y)
    print("Accuracy9 (ET-A, pcaX):",acc9)
    
    # =============================================================================
     
    print('==================================================')
    #STILL NOT SO GOOD - LETS TRY TO PREDICT NOT THE FINAL RESULT BUT ONLY IF STUDENT PASSED (SCORE>=10) OR NOT
    print("DATASET PREDICTIONS PASSED-FAIL")
    array = dataset.values
    Y = array[:,32]
    Y_new = binarizePredictedValues(Y)

    # RANDOM FORESTS
    #1: predict only pass/fail from whole data with Random Forests
    acc1 = predictionRandomForests(array[:,0:32], Y_new)
    print("Accuracy1 (RF-B, allData):",acc1)
    #2:  predict only pass/fail but only with 3 main components - G1, G2, abscences with Random Forests
    acc2 = predictionRandomForests(array[:,29:32], Y_new)
    print("Accuracy2 (RF-B, main3comp):",acc2)    
    #3:  predict only pass/fail with data transformed from PCA with Random Forests
    acc3 = predictionRandomForests(transformed_X, Y_new)
    print("Accuracy3 (RF-B, pcaX):",acc3)

    # DECISION TREES
    #4: predict only pass/fail from whole data with Decision Trees
    acc4 = predictionDecisionTree(array[:,0:32], Y_new)
    print("Accuracy4 (DT-B, allData):",acc4)
    #5:  predict only pass/fail but only with 3 main components - G1, G2, abscences with Decision Trees
    acc5 = predictionDecisionTree(array[:,29:32], Y_new)
    print("Accuracy5 (DT-B, main3comp):",acc5)    
    #6:  predict only pass/fail with data transformed from PCA with Decision Trees
    acc6 = predictionDecisionTree(transformed_X, Y_new)
    print("Accuracy6 (DT-B, pcaX):",acc6)
    
    # EXTRA TREES
    #7: predict only pass/fail from whole data with Extra Trees
    acc7 = predictionExtraTrees(array[:,0:32], Y_new)
    print("Accuracy7 (ET-B, allData):",acc7)
    #8:  predict only pass/fail but only with 3 main components - G1, G2, abscences with Extra Trees
    acc8 = predictionExtraTrees(array[:,29:32], Y_new)
    print("Accuracy8 (ET-B, main3comp):",acc8)    
    #9:  predict only pass/fail with data transformed from PCA with Extra Trees
    acc9 = predictionExtraTrees(transformed_X, Y_new)
    print("Accuracy9 (ET-B, pcaX):",acc9)
    
    # =============================================================================
    # =============================================================================
    print('==================================================')
    print("DATASET PREDICTIONS WITHOUT GRADES")
    # This is nice, but it would be more interesting to have predictions of final grade without other grades G1 a G2
    array = dataset.values
    X = array[:,0:30]
    Y = array[:,32]
    
    #1 with Random Forests 
    model = featureExtractionRandomForestClassifier(X,Y)
    feature_names = dataset.columns.values[:-3]
    visualization(dataset, model, feature_names, "Important features - Random forest")
    #2 with Decision Trees
    model = featureExtractionDecisionTrees(X,Y)
    visualization(dataset, model, feature_names, "Important features - Decision tree")
    #3 with Extra Trees
    model = featureExtractionExtraTreesClassifier(X,Y)
    visualization(dataset, model, feature_names, "Important features - Extra tree")
    
    #PRINCIPAL COMPONENT ANALYSIS
    transformed_X = pca(dataset, 1, 30)
        
    #1: RandomForest to predict resultant score on whole dataset
    acc1 = predictionRandomForests(X, Y)
    print("Accuracy1 (RF-A, noGradesAll:",acc1)    
    #2: now lets do the same but with data transformed from PCA
    acc2 = predictionRandomForests(transformed_X, Y)
    print("Accuracy2 (RF-A, noGradesPcaX):",acc2)
    
    #3: DecisionTree to predict resultant score on whole dataset
    acc3 = predictionDecisionTree(X, Y)
    print("Accuracy3 (DT-A, noGradesAll):",acc3)   
    #4: now lets do the same but with data transformed from PCA
    acc4 = predictionDecisionTree(transformed_X, Y)
    print("Accuracy4 (DT-A, noGradesPcaX):",acc4)
    
    #5: ExtraTrees to predict resultant score on whole dataset
    acc5 = predictionExtraTrees(X, Y)
    print("Accuracy5 (ET-A, noGradesAll):",acc5)    
    #6: now lets do the same but with data transformed from PCA
    acc6 = predictionExtraTrees(transformed_X, Y)
    print("Accuracy6 (ET-A, noGradesPcaX):",acc6)
     
    #PREDICT NOT THE FINAL RESULT BUT ONLY IF STUDENT PASSED (SCORE>=10) OR NOT
    array = dataset.values
    X = array[:,0:30]
    Y = array[:,32]
    Y_new = binarizePredictedValues(Y)

    # RANDOM FORESTS
    #1: predict only pass/fail from whole data with Random Forests
    acc1 = predictionRandomForests(array[:,0:30], Y_new)
    print("Accuracy1 (RF-B, noGradesAll):",acc2) 
    #2:  predict only pass/fail with data transformed from PCA with Random Forests
    acc2 = predictionRandomForests(transformed_X, Y_new)
    print("Accuracy2 (RF-B, noGradesPcaX):",acc2)

    # DECISION TREES
    #3: predict only pass/fail from whole data with Decision Trees
    acc3 = predictionDecisionTree(array[:,0:30], Y_new)
    print("Accuracy3 (DT-B, noGradesAll):",acc3)   
    #4:  predict only pass/fail with data transformed from PCA with Decision Trees
    acc4 = predictionDecisionTree(transformed_X, Y_new)
    print("Accuracy4 (DT-B, noGradesPcaX):",acc4)
    
    # EXTRA TREES
    #5: predict only pass/fail from whole data with Extra Trees
    acc5 = predictionExtraTrees(array[:,0:30], Y_new)
    print("Accuracy5 (ET-B, noGradesAll):",acc5)  
    #6:  predict only pass/fail with data transformed from PCA with Extra Trees
    acc6 = predictionExtraTrees(transformed_X, Y_new)
    print("Accuracy6 (ET-B, noGradesPcaX):",acc6)
    
    
if __name__ == '__main__':
    main()




#TODO: useful resources (DELETE THEM BEFORE SUBMITING)
#https://machinelearningmastery.com/feature-selection-machine-learning-python/
#https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706
#https://www.datacamp.com/community/tutorials/random-forests-classifier-python









