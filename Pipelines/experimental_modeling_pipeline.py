#!/usr/bin/env python
'''This file contains two classes who facilitate the data preprocessing and subsequent modeling of selected files from the UK biobank
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import statistics as stat
from sklearn.impute import KNNImputer

__author__ = "Keimpe Dijkstra"
__credits__ = ["Stefan Wijtsma", "Lujein Alsheik"]
__license__ = "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"
__version__ = "1.0.1"
__maintainer__ = "Keimpe Dijkstra"
__email__ = "k.dijkstra@labonovum.com"

class Experimental_modeling_pipeline():
    '''This class is a general modeling class based on the sklearn software package.
    After an instance of the class is created functions can be used seperately,
    or the .pipeline method can be called upon which takes several arguments.

    Methods
    -------
    train_test_data(df, test_size, datacolumns, datalabel)
        This function splits the data for testing and training.
    logistic_regression(x_train, y_train, multiclass)
        This function creates a logistic regression model
    model_predict(model, x_test, multiclass)
        This function handles predictions using a given model
    zeror(df, column)
        This function returns the results from a zeroR algorithm
    zeror_score(vc)
        This function calculates the score from the zeroR algorithm
    auc(y_test, predictions)
        This function calculates the AUC of the ROC
    auc_multiclass(y_test, predictions)
        This function calculates the AUC of the ROC for a multiclass problem
    naive_bayes(x_train, y_train)
        This function creates a naíve bayes model
    barchart_evaluation(auc, accuracy, models_str)
        This function visualizes the AUC and accuracy of the models in a bar chart
    random_forest(x_train, y_train)
        This function creates a random forest model
    confusion_matrix(y_test,predicted)
        This function visualizes the confusion matrix of a binary solver
    multilabel_confusion_matrix(y_test, predicted)
        This function visualizes the confusion matrix of a multiclass solver
    f1_score_out(y_test, prediction, modelname, multiclass)
        This function returns the F1 score
    kfold_crossvalidation(model, splits, x, y, multiclass)
        This function returns scores for a k-fold crassvalidation model 
    impute_mean(df, datacolumns)
        This function imputes given dataframe columns by their mean
    impute_median(df, datacolumns)
        This function imputes given dataframe columns by their median
    impute_knn(df, datacolumns, neigbours,knn_weights)
        This function imputes given dataframe columns using the k-nearest neighbor algorithm
    impute_mean_labelbased(df, datacolumns, datalabel)
        This function imputes given dataframe columns by the mean of their respective class 
    impute_median_labelbased(df, datacolumns, datalabel)
        This function imputes given dataframe columns by the median of their respective class 
    pipeline( df ,datalabel, datacolumns, test_size, multiclass, splits, impute, neighbours, knn_weights):
        This function acts as a automised pipeline calling needed functions based on input
    '''

    def __init__(self):
        '''Initialize object, this does not take any arguments
        '''

        self.models_str =  ["zeroR", "Logistic regression", "Naive bayes", "Random forest"]

    def train_test_data(self, df, test_size, datacolumns, datalabel):
        '''This function splits the data for testing and training.
        
        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be split
        test_size : int
            Size in percentage of the test group
        Datacolumns : array
            List containing the column names for the data to be used
        datalabel : str
            String indicating the column used as label

        Returns
        -------
        x_train : pandas dataframe object
            Contains training data
        x_test : pandas dataframe object
            Contains testing data
        y_train : array
            Contains training class labels
        y_test : array
            Contains testing class labels
        '''

        x_train, x_test, y_train, y_test = train_test_split(df[datacolumns], df[datalabel], test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def logistic_regression(self, x_train, y_train, multiclass):
        '''This function creates a logistic regression model

        Parameters
        ----------
        x_train : pandas dataframe object
            Data on which the model should be trained
        y_train : array
            List containing class labels
        multiclass : int
            Binary indicating whether it involves a multiclass problem

        Returns
        -------
        logisticRegr : sklearn LogisticRegression object
            Logistic resgression model
        '''

        if multiclass == 1:
            logisticRegr = LogisticRegression(multi_class='multinomial', max_iter=1000000000)
        else:
            logisticRegr = LogisticRegression(max_iter=1000000000)
        logisticRegr.fit(x_train, y_train)
        return logisticRegr
    
    def model_predict(self, model, x_test, multiclass):
        '''This function handles predictions using a given model

        Parameters
        ----------
        model : sklearn model object
        x_test : pandas dataframe object
            Data with whom the model should be tested
        multiclass : int
            Binary indicating whether it involves a multiclass problem

        Returns
        -------
        Predictions : array
            List with class predictions
        Predictions_prob : array
            List with probabilities of the predictions
        '''

        if multiclass == 1:
            predictions = model.predict(x_test) #predict_proba
            predictions_prob = model.predict_proba(x_test)
            return predictions, predictions_prob
        else:
            predictions = model.predict(x_test)
            predictions_prob = model.predict_proba(x_test)
            return predictions, predictions_prob
    
    def model_score(self, model, x_test, y_test):
        '''This function handles scores using a given model and testing data

        Parameters
        ----------
        model : sklearn model object
        x_test : pandas dataframe object
            Data with whom the model should be tested
        y_test : array
            Contains testing class labels
        
        Returns
        -------
        score : sklearn score object
        '''
        score = model.score(x_test, y_test)
        return score
    
    def zeror(self, df, column):
        '''This function returns the results from a zeroR algorithm

        Parameters
        ----------
        df : pandas dataframe object
        column : str
            String indicating the column used as label

        Returns
        -------
        vc : value counts object
        '''

        vc = df[column].value_counts()
        return vc
    
    def zeror_score(self, vc):
        '''This function calculates the score from the zeroR algorithm

        Parameters
        ----------
        vc : value counts object

        Returns
        -------
        Percentage correct : float
        '''
        return max(vc) / sum(vc)
        
    def auc(self, y_test, predictions):
        '''This function calculates the AUC of the ROC

        Parameters
        ----------
        y_test : array
            Contains testing class labels
        Predictions : array
            List with class predictions

        Returns
        -------
        auc score : float
        '''

        return metrics.roc_auc_score(y_test, predictions)
    
    def auc_multiclass(self, y_test, predictions):
        '''This function calculates the AUC of the ROC for a multiclass problem

        Parameters
        ----------
        y_test : array
            Contains testing class labels
        Predictions : array
            List with class predictions

        Returns
        -------
        auc score : float
        '''

        return metrics.roc_auc_score(y_test, predictions, multi_class='ovo')
    
    def naive_bayes(self, x_train, y_train):
        '''This function creates a naíve bayes model

        Parameters
        ----------
        x_train : pandas dataframe object
            Data on which the model should be trained
        y_train : array
            List containing class labels
        
        Returns
        -------
        model : sklearn Guassian naïve bayes object
        '''

        model = GaussianNB()
        model.fit(x_train, y_train)
        return model
    
    def barchart_evaluation(self, auc, accuracy, models_str):
        '''This function visualizes the AUC and accuracy of the models in a bar chart

        Parameters
        ----------
        auc : array
            List with auc scores
        accuracy : array
            List with accuracy scores
        models_str : list
            List with names of models as string
        '''

        X_axis = np.arange(len(models_str)) 

        plt.bar(X_axis - 0.2, auc, 0.4, label = 'auc') 
        plt.bar(X_axis + 0.2, accuracy, 0.4, label = 'accuracy') 

        plt.xticks(X_axis, models_str) 
        plt.xlabel("Evaluations") 
        plt.ylabel("Score") 
        plt.title("Score of various models") 
        plt.legend() 
        plt.show() 

    def random_forest(self, x_train, y_train):
        '''This function creates a random forest model

        Parameters
        ----------
        x_train : pandas dataframe object
            Data on which the model should be trained
        y_train : array
            List containing class labels
        
        Returns
        -------
        model : sklearn random forest object
        '''

        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        return model
    
    def confusion_matrix(self,y_test,predicted ):
        '''This function visualizes the confusion matrix of a binary solver

        Parameters
        ----------
        y_test : array
            Contains testing class labels
        predicted : array
            List with class predictions
        '''

        confusion_matrix = metrics.confusion_matrix(y_test, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

        cm_display.plot()
        plt.show()

    def multilabel_confusion_matrix(self,y_test,predicted ):
        '''This function visualizes the confusion matrix of a multiclass solver

        Parameters
        ----------
        y_test : array
            Contains testing class labels
        predicted : array
            List with class predictions
        '''

        cm = metrics.multilabel_confusion_matrix(y_test,predicted )
        for c in cm:
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c)
            cm_display.plot()
            plt.show()

    def f1_score_out(self, y_test, prediction, modelname, multiclass):
        '''This function returns the F1 score
        
        Parameters
        ----------
        y_test : array
            Contains testing class labels
        prediction : array
            List with class predictions
        modelname : str
            Nmae of model for which the F1 score is calculated
        multiclass : int
            Binary indicating whether it involves a multiclass problem
        '''

        if multiclass == 0:
            print("f1 score ", modelname, ": ", metrics.f1_score(y_test, prediction))
        else:
            print("f1 score ", modelname, ": ", metrics.f1_score(y_test, prediction, average="micro"))

    def kfold_crossvalidation(self, model, splits, x, y, multiclass):
        '''This function returns scores for a k-fold crassvalidation model 

        Parameters
        ----------
        model : sklearn model object
        splits : int
            Number of times the data should be split
        x: pandas dataframe object
            Data on which the model should be trained and tested
        y : array
            List containing class labels 
        multiclass : int
            Binary indicating whether it involves a multiclass problem

        Returns
        -------
        score : sklearn score object
        '''

        if multiclass == 0:
            s = ["accuracy", "f1", "roc_auc"]
        if multiclass ==1:
            s = ["accuracy", "f1_macro", "roc_auc_ovr", "f1_micro"]
        scores = cross_validate(estimator=model, X=x, y=y, cv=splits, scoring=s)
        return scores
        
    def impute_mean(self, df, datacolumns):
        '''This function imputes given dataframe columns by their mean

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be imputed
        Datacolumns : array
            List containing the column names for the data to be used

        Returns
        -------
        df : pandas dataframe object
            Imputed dataframe
        '''

        for c in datacolumns:
            if len(df[c].unique().tolist()) == 2:
                df[c].fillna(df[c].mode(), inplace = True)
            else:
                df[c].fillna(df[c].mean(), inplace = True)
        return df    
    
    def impute_median(self, df, datacolumns):
        '''This function imputes given dataframe columns by their median

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be imputed
        Datacolumns : array
            List containing the column names for the data to be used

        Returns
        -------
        df : pandas dataframe object
            Imputed dataframe
        '''

        for c in datacolumns:
            if len(df[c].unique().tolist()) == 2:
                df[c].fillna(df[c].mode(), inplace = True)
            else:
                df[c].fillna(df[c].median(), inplace = True)
        return df
    
    def impute_knn(self, df, datacolumns, neigbours,knn_weights):
        '''This function imputes given dataframe columns using the k-nearest neighbor algorithm

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be imputed
        Datacolumns : array
            List containing the column names for the data to be used
        neighbours : int
            Number of datapoints used for estimation 
        knn_weights : str
            Weight function used in prediction
            
        Returns
        -------
        df : pandas dataframe object
            Imputed dataframe
        '''

        imputer = KNNImputer(n_neighbors=neigbours, weights=knn_weights)
        df = pd.DataFrame(imputer.fit_transform(df[datacolumns]),columns = datacolumns)
        return df
    
    def impute_mean_labelbased(self, df, datacolumns, datalabel):
        '''This function imputes given dataframe columns by the mean of their respective class 

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be imputed
        Datacolumns : array
            List containing the column names for the data to be used
        datalabel : str
            String indicating the column used as label

        Returns
        -------
        df : pandas dataframe object
            Imputed dataframe
        '''

        df[datacolumns] = df[datacolumns].fillna(df.groupby(datalabel)[datacolumns].transform('mean'))
        return df
    
    def impute_median_labelbased(self, df, datacolumns, datalabel):
        '''This function imputes given dataframe columns by the median of their respective class 

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be imputed
        Datacolumns : array
            List containing the column names for the data to be used
        datalabel : str
            String indicating the column used as label
            
        Returns
        -------
        df : pandas dataframe object
            Imputed dataframe
        '''

        df[datacolumns] = df[datacolumns].fillna(df.groupby(datalabel)[datacolumns].transform('median'))
        return df

    def pipeline(self, df ,datalabel, datacolumns, test_size, multiclass=0, splits=0, impute="", neighbours=5, knn_weights="distance"):
        '''This function acts as a automised pipeline calling needed functions based on input

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be used
        Datacolumns : array
            List containing the column names for the data to be used
        datalabel : str
            String indicating the column used as label
        test_size : int
            Size in percentage of the test group
        multiclass : int
            Binary indicating whether it involves a multiclass problem (default=0)
        splits : int
            Number of times the data should be split (default=0)
        impute : str
            String indicating method of imputation (mean / median / knn / mean_labelbased / median_labelbased / default="")
        neighbours: int
            Number of datapoints used for estimation in knn imputation (default=5)
        knn_weights : str
            Weight function used in knn prediction (default="distance")
        '''
        
        #Impution selection
        if impute == "mean":
            df = self.impute_mean(df=df, datacolumns=datacolumns)
        if impute == "median":
            df = self.impute_median(df=df, datacolumns=datacolumns)
        if impute == "knn":
            df_id = df["Participant ID"]
            df_label = df[datalabel]
            df = self.impute_knn(df=df[1:], datacolumns=datacolumns, neigbours=neighbours, knn_weights=knn_weights)
            df = pd.concat([df_id, df], axis=1)
            df = pd.concat([df, df_label], axis=1)
        if impute == "mean_labelbased":
            df = self.impute_mean_labelbased(df=df, datacolumns=datacolumns, datalabel=datalabel)
        if impute == "median_labelbased":
            df = self.impute_median_labelbased(df=df, datacolumns=datacolumns, datalabel=datalabel)
        
        #Return number of missing values and drop any missing values
        print("Number of missing values: \n", df.isna().sum())
        df = df.dropna()
        print("Dataframe dimensions: ", df.shape)

        #Calculate zeroR
        zeror = self.zeror(df, datalabel)
        if splits ==0:     
            #Construct models with no cross-validation   
            x_train, x_test, y_train, y_test = self.train_test_data(df, test_size, datacolumns, datalabel)
            logistic_regression = self.logistic_regression(x_train, y_train, multiclass)
            naive_bayes = self.naive_bayes(x_train, y_train)
            random_forest = self.random_forest(x_train, y_train)

            #Predict using constructed models
            lg_predict, lg_predict_prob = self.model_predict(logistic_regression, x_test, multiclass)
            nb_predict, nb_predict_prob = self.model_predict(naive_bayes, x_test, multiclass)
            rf_predict, rf_predict_prob = self.model_predict(random_forest, x_test, multiclass)
            
            #Retrieve scores of models
            if multiclass == 1:
                auc = [0.5, self.auc_multiclass(y_test, lg_predict_prob), self.auc_multiclass(y_test, nb_predict_prob), self.auc_multiclass(y_test, rf_predict_prob) ]
            else:
                auc = [0.5, self.auc(y_test, lg_predict), self.auc(y_test, nb_predict), self.auc(y_test, rf_predict) ]

            acc = [self.zeror_score(zeror), self.model_score(logistic_regression, x_test, y_test)
                , self.model_score(naive_bayes, x_test, y_test), self.model_score(naive_bayes, x_test, y_test)]
            
            #Plot and print scores
            self.barchart_evaluation(auc, accuracy=acc, models_str=self.models_str )
            
            self.f1_score_out(y_test, lg_predict, self.models_str[1], multiclass)
            self.f1_score_out(y_test, nb_predict, self.models_str[2], multiclass)
            self.f1_score_out(y_test, rf_predict, self.models_str[3], multiclass)
            
            print("auc: ", auc)

            if multiclass == 0:
                self.confusion_matrix(y_test, lg_predict)
                self.confusion_matrix(y_test, nb_predict)
                self.confusion_matrix(y_test, rf_predict)
            
            elif multiclass ==1:
                self.multilabel_confusion_matrix(y_test, lg_predict)
                self.multilabel_confusion_matrix(y_test, nb_predict)
                self.multilabel_confusion_matrix(y_test, rf_predict)

        if splits > 0:
            #Seperate data and class labels
            x = df[datacolumns]
            y = df[datalabel]
            
            #Construct models using cross-validation
            if multiclass == 1:
                scores_lr = self.kfold_crossvalidation(splits=splits, x=x, y=y, model=LogisticRegression(multi_class='multinomial', max_iter=1000000000), multiclass=multiclass)
                
            if multiclass == 0:
                scores_lr = self.kfold_crossvalidation(splits=splits, x=x, y=y, model=LogisticRegression(max_iter=1000000000),multiclass=multiclass)
            scores_nb = self.kfold_crossvalidation(splits=splits, x=x, y=y, model=GaussianNB(), multiclass=multiclass)
            scores_rf = self.kfold_crossvalidation(splits=splits, x=x, y=y, model=RandomForestClassifier(), multiclass=multiclass)
            
            #Return and visualize scores
            if multiclass == 0:
                auc = [self.zeror_score(zeror), stat.mean(scores_lr["test_roc_auc"]),stat.mean(scores_nb["test_roc_auc"]), stat.mean(scores_rf["test_roc_auc"])]
                acc = [self.zeror_score(zeror), stat.mean(scores_lr["test_accuracy"]),stat.mean(scores_nb["test_accuracy"]), stat.mean(scores_rf["test_accuracy"])]
                self.barchart_evaluation(auc, accuracy=acc, models_str=self.models_str )
                print("F1: ", stat.mean(scores_lr["test_f1"]),stat.mean(scores_nb["test_f1"]), stat.mean(scores_rf["test_f1"]))
            if multiclass == 1:
                auc = [self.zeror_score(zeror), stat.mean(scores_lr["test_roc_auc_ovr"]),stat.mean(scores_nb["test_roc_auc_ovr"]), stat.mean(scores_rf["test_roc_auc_ovr"])]
                acc = [self.zeror_score(zeror), stat.mean(scores_lr["test_accuracy"]),stat.mean(scores_nb["test_accuracy"]), stat.mean(scores_rf["test_accuracy"])]
                self.barchart_evaluation(auc, accuracy=acc, models_str=self.models_str )
                print("F1_macro: ", stat.mean(scores_lr["test_f1_macro"]),stat.mean(scores_nb["test_f1_macro"]), stat.mean(scores_rf["test_f1_macro"]))
                print("F1_micro: ", stat.mean(scores_lr["test_f1_micro"]),stat.mean(scores_nb["test_f1_micro"]), stat.mean(scores_rf["test_f1_micro"]))
                
            print("auc: ", auc)
            print("accuracy: ", acc)
            
    
class DiabetesPreprocessing():
    '''This class handles the preprocessing of selected files obtained from the UK biobank
    '''

    def __init__(self, wd):
        '''
        Init function requires one argument:
            wd: str
                Working directory 
        '''
        self.first_occurence = wd + "DM_first_occurence_dates.csv"
        self.family_history = wd + "Family_history.csv"
        self.attendance = wd + "Dates_attending_assessment_centers_participant.csv"
        self.demographics = wd + "Demographics.csv"
        self.bodymeasures = wd + "Body_measures.csv"
        self.blood_biomarkers = wd + "Blood_biomarkers.csv"
        self.bloodpressure = wd + "Blood_pressure_raw.csv"
        self.urine_biomarkers = wd + "Urine_biomarkers.csv"
        self.medical_conditions = wd + "Medical_conditions.csv"
        self.alcohol = wd + "Alcohol.csv"
        self.physical_activity = wd + "Physical_activity.csv"
        self.sleep = wd + "Sleep.csv"
        self.smoking = wd + "Smoking.csv"
        self.white_bloodcell = wd + "Blood_biomarkers_3.csv"

    def pre_first_occurence_labels(self):
        '''
        Returns dataframe with two labels:
            1) first_occurence_binary: label which indicates whether a first occurence of diabetes has happend (no=0, yes=1)
            2) first_occurence_tertiary: label which indicates whether a first occurence has happend, is going to happen,
                                         or doesn't happen (0: did not happen, 1: did happen, 3: is going to happen)

        Uses random undersampling
        ''' 
        df = pd.read_csv(self.first_occurence)
        df["first_occurence"] = df[df.columns[1:]].apply(self.earliest_date, axis=1) #Perform the earliest_date function
        df["first_occurence_binary"] = df["first_occurence"].apply(self.nan_to_binary) #Perform the nan_to_binary function
        df = df.drop(df.columns[1:-1], axis=1) #drop non-important columns

        df_dm = df[df["first_occurence_binary"] == 1] #Get all the rows with a first occurence
        df_nodm = df[df["first_occurence_binary"] == 0].dropna() #Get all rows without first occurence 
        df_nodm_sampled = df_nodm.sample(n=df_dm.shape[0], axis=0) #Undersample the two dataframes
        frames = [df_dm, df_nodm_sampled]
        df = pd.concat(frames, axis=0) #Combine the aforementioned datasets

        att = pd.read_csv(self.attendance)
        fo =  pd.read_csv(self.first_occurence)
        fo["first_occurence"] = fo[fo.columns[1:]].apply(self.earliest_date, axis=1)
        att = att[["Participant ID", "Date of attending assessment centre | Instance 0"]] #Select columns
        fo = fo[["Participant ID", "first_occurence"]]
        fo = fo.merge(att, on = "Participant ID")
        fo["first_occurence_tertiary"] = fo.apply(self.nan_to_tertiary, axis=1) #Apply nan_to_tertiary function
        fo = fo[["Participant ID", "first_occurence_tertiary"]]
        df = df.merge(fo, on = "Participant ID")

        return df
    
    def pre_family_history(self):
        '''
        Returns dataframe with three rows indicating the presence of diabetes in father, mother and sibling binary
        '''
        fh = pd.read_csv(self.family_history)
        fh = self.merge_noneoftheabove(fh)
        fh = self.remove_redundant_noneoftheabove(fh)
        dm, other, uncertain = self.diagnosis_including_diabetes(fh, fh.columns[1:])
        #Create new columns indicating whether a familymember suffers from diabetes (1:yes,0:no,3:uncertain)
        fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )] = fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )].replace(to_replace=dm, value=1)

        fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )] = fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )].replace(to_replace=other, value=0)

        fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )] = fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )].replace(to_replace=uncertain, value=3)

        conditions = [ (fh["Illnesses of father | Instance 0"] == 1),
                    (fh["Illnesses of father | Instance 1"] == 1),
                    (fh["Illnesses of father | Instance 2"] == 1),
                    (fh["Illnesses of father | Instance 3"] == 1),
                    (fh["Illnesses of father | Instance 0"] == 3),
                    (fh["Illnesses of father | Instance 1"] == 3),
                    (fh["Illnesses of father | Instance 2"] == 3),
                    (fh["Illnesses of father | Instance 3"] == 3),
                    (fh["Illnesses of father | Instance 0"] == 0),
                    (fh["Illnesses of father | Instance 1"] == 0),
                    (fh["Illnesses of father | Instance 2"] == 0),
                    (fh["Illnesses of father | Instance 3"] == 0)
                    ]

        values = [1,1,1,1,3,3,3,3,0,0,0,0]
        fh["Illnesses of father"] = np.select(conditions, values)

        conditions = [
                    (fh["Illnesses of mother | Instance 0"] == 1),
                    (fh["Illnesses of mother | Instance 1"] == 1),
                    (fh["Illnesses of mother | Instance 2"] == 1),
                    (fh["Illnesses of mother | Instance 3"] == 1),
                    (fh["Illnesses of mother | Instance 0"] == 3),
                    (fh["Illnesses of mother | Instance 1"] == 3),
                    (fh["Illnesses of mother | Instance 2"] == 3),
                    (fh["Illnesses of mother | Instance 3"] == 3),
                    (fh["Illnesses of mother | Instance 0"] == 0),
                    (fh["Illnesses of mother | Instance 1"] == 0),
                    (fh["Illnesses of mother | Instance 2"] == 0),
                    (fh["Illnesses of mother | Instance 3"] == 0)]
        fh["Illnesses of mother"] = np.select(conditions, values)

        conditions = [
                    (fh["Illnesses of siblings | Instance 0"] == 1),
                    (fh["Illnesses of siblings | Instance 1"] == 1),
                    (fh["Illnesses of siblings | Instance 2"] == 1),
                    (fh["Illnesses of siblings | Instance 3"] == 1),
                    (fh["Illnesses of siblings | Instance 0"] == 3),
                    (fh["Illnesses of siblings | Instance 1"] == 3),
                    (fh["Illnesses of siblings | Instance 2"] == 3),
                    (fh["Illnesses of siblings | Instance 3"] == 3),
                    (fh["Illnesses of siblings | Instance 0"] == 0),
                    (fh["Illnesses of siblings | Instance 1"] == 0),
                    (fh["Illnesses of siblings | Instance 2"] == 0),
                    (fh["Illnesses of siblings | Instance 3"] == 0)]
        fh["Illnesses of siblings"] = np.select(conditions, values)

        fh = fh.drop(fh.columns[1:13], axis=1)
        fh = fh[1:].replace(3,0)
        return fh
    
    def pre_demographics_ethnicity(self):
        '''
        Returns a dataframe with one hot encodings for six main ethnicities
        '''
        d = pd.read_csv(self.demographics)
        d = self.ethnicity_regrouping(d, "Ethnic background | Instance 0")
        d = d[["Participant ID", "Ethnic background | Instance 0"]]
        one_hot = pd.get_dummies(d["Ethnic background | Instance 0"])
        d = pd.concat([d, one_hot], axis=1)
        d = d[['Participant ID',
               'Asian',
                'Black',
                'Chinese',
                'Mixed',
                'Other',
                'White']]
        return d
    
    def pre_bodymeasures(self):
        '''
        Returns a dataframe with various columns on different type of body measures 
        '''
        bm = pd.read_csv(self.bodymeasures)
        first_and_general_instances = [i for i in list(bm.columns) if not re.search("1|2|3", i)]#Make a list with first and general instances
        remove_columns = [ i for i in bm.columns if i not in first_and_general_instances ]#Construct a list with all second, third and fouth instances
        bm = bm.drop(remove_columns, axis=1)#Remove instances
        bm["fmi"] = bm["Whole body fat mass | Instance 0"] / ((bm["Standing height | Instance 0"] /100)* (bm["Standing height | Instance 0"]/100))#Create fat mass index column
        bm = bm[["Participant ID","fmi",
                "Body mass index (BMI) | Instance 0",
                "Body fat percentage | Instance 0",
                "Waist circumference | Instance 0",
                "Weight | Instance 0",
                "Hip circumference | Instance 0",
                "Whole body fat mass | Instance 0",
                "Basal metabolic rate | Instance 0",
                "Trunk fat percentage | Instance 0",
                "Arm fat percentage (left) | Instance 0",
                "Leg fat percentage (left) | Instance 0"
                ]]
        return bm
    
    def pre_blood_biomarker(self, dem_basic):
        '''
        Returns dataframe with blood biomarker columns 
        '''
        bbm = pd.read_csv(self.blood_biomarkers)
        first_and_general_instances = [i for i in list(bbm.columns) if not re.search("Instance 1|Instance 2|Instance 3", i)]
        remove_columns = [ i for i in bbm.columns if i not in first_and_general_instances ]
        bbm = bbm.drop(remove_columns, axis=1)
        bbm["trigl_hdl_ratio"] = (bbm["Triglycerides | Instance 0"] * 88.57 ) / (bbm["HDL cholesterol | Instance 0"] *38.67)#Add triglcerides to hdl ratio for estimation of molecule size
        bbm["apob_apoa_ratio"] = bbm["Apolipoprotein B | Instance 0"] / bbm["Apolipoprotein A | Instance 0"] #Construct apob / apoa ratio 
        bbm = bbm.drop(columns=['Glycated haemoglobin (HbA1c) assay date | Instance 0'])
        bbm = bbm.merge(dem_basic , on="Participant ID")
        bbm["gfr"] = bbm.apply(self.gfr, axis=1)
        return bbm
    
    def pre_blood_pressure(self):
        '''
        Returns a dataframe with diastolic and systolic blood pressure
        '''
        bp = pd.read_csv(self.bloodpressure)
        bp = bp[['Participant ID',
        'Diastolic blood pressure, automated reading | Instance 0 | Array 0',
        'Diastolic blood pressure, automated reading | Instance 0 | Array 1',
        'Systolic blood pressure, automated reading | Instance 0 | Array 0',
        'Systolic blood pressure, automated reading | Instance 0 | Array 1']]

        bp['Diastolic blood pressure'] = bp[['Diastolic blood pressure, automated reading | Instance 0 | Array 0',
        'Diastolic blood pressure, automated reading | Instance 0 | Array 1']].mean(axis=1)
        bp['Systolic blood pressure'] = bp[['Systolic blood pressure, automated reading | Instance 0 | Array 0',
        'Systolic blood pressure, automated reading | Instance 0 | Array 1']].mean(axis=1)

        return bp[["Participant ID",'Diastolic blood pressure', 'Systolic blood pressure' ]]
    
    def pre_demographics_basic(self):
        '''
        Returns a dataframe with age and sex columns
        '''
        d = pd.read_csv(self.demographics)
        d = d[['Participant ID', 'Age at recruitment', 'Sex']]
        d["Sex_binary"] = d[["Sex"]].apply(self.sex_to_binary, axis=1)
        d = d.drop(["Sex"], axis=1)
        return d
    
    def pre_urine_biomarkers(self):
        '''
        Returns a dataframe with urine biomarkers
        '''
        bm = pd.read_csv(self.urine_biomarkers, low_memory=False)
        first_and_general_instances = [i for i in list(bm.columns) if not re.search("Instance 1|Instance 2|Instance 3|flag", i)]
        remove_columns = [ i for i in bm.columns if i not in first_and_general_instances ]
        bm = bm.drop(remove_columns, axis=1)
        bm["Creatinine (enzymatic) in urine | Instance 0"] = bm["Creatinine (enzymatic) in urine | Instance 0"] / 1000000 * 113.12 #convert creatine in urine from mmol/L to mg/dL
        bm["albumin_creatine_ratio"] = bm["Microalbumin in urine | Instance 0"] / bm["Creatinine (enzymatic) in urine | Instance 0"] #add albumine creatine ratio
        return bm
    
    def pre_medications(self):
        '''
        Returns a dataframe with one hot encoding for medications for cholesterol lowering; insulin; blood pressure
        '''
        mc = pd.read_csv(self.medical_conditions, low_memory=False)
        mc["Cholesterol_lowering_medication"] = mc[["Medication for cholesterol, blood pressure or diabetes | Instance 0", 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones | Instance 0']].apply(self.cholesterol_check, axis=1)
        mc["Insulin"] = mc[["Medication for cholesterol, blood pressure or diabetes | Instance 0", 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones | Instance 0']].apply(self.insulin_check, axis=1)
        mc["Blood_pressure"] = mc[["Medication for cholesterol, blood pressure or diabetes | Instance 0", 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones | Instance 0']].apply(self.Blood_pressure_check, axis=1)
        return mc[["Participant ID","Cholesterol_lowering_medication", "Insulin", "Blood_pressure"]]

    def pre_alcohol(self): 
        '''
        Returns one hot encoding for selected rates of alcohol consumption
        '''
        alc = pd.read_csv(self.alcohol,low_memory=False)
        alc = pd.get_dummies(alc, columns=["Alcohol intake frequency. | Instance 0"])
        alc = alc[['Participant ID', 
            'Alcohol intake frequency. | Instance 0_Daily or almost daily',
            'Alcohol intake frequency. | Instance 0_Never',
            'Alcohol intake frequency. | Instance 0_Once or twice a week',
            'Alcohol intake frequency. | Instance 0_One to three times a month',
            'Alcohol intake frequency. | Instance 0_Special occasions only',
            'Alcohol intake frequency. | Instance 0_Three or four times a week']]
        return alc
    
    def pre_family_history_other(self):
        '''
        Returns a dataframe with selected diseases in direct family(onehot)
        '''
        fh = pd.read_csv(self.family_history)

        fh["father_stroke"] = fh[["Illnesses of father | Instance 0"]].apply(self.stroke_check, axis=1)
        fh["mother_stroke"] = fh[["Illnesses of mother | Instance 0"]].apply(self.stroke_check, axis=1)
        fh["sibling_stroke"] = fh[["Illnesses of siblings | Instance 0"]].apply(self.stroke_check, axis=1)

        fh["father_alzheimer"] = fh[["Illnesses of father | Instance 0"]].apply(self.alzheimer_check, axis=1)
        fh["mother_alzheimer"] = fh[["Illnesses of mother | Instance 0"]].apply(self.alzheimer_check, axis=1)
        fh["sibling_alzheimer"] = fh[["Illnesses of siblings | Instance 0"]].apply(self.alzheimer_check, axis=1)

        fh["father_bloodpressure"] = fh[["Illnesses of father | Instance 0"]].apply(self.bloodpressure_check, axis=1)
        fh["mother_bloodpressure"] = fh[["Illnesses of mother | Instance 0"]].apply(self.bloodpressure_check, axis=1)
        fh["sibling_bloodpressure"] = fh[["Illnesses of siblings | Instance 0"]].apply(self.bloodpressure_check, axis=1)

        fh["father_heart"] = fh[["Illnesses of father | Instance 0"]].apply(self.heart_check, axis=1)
        fh["mother_heart"] = fh[["Illnesses of mother | Instance 0"]].apply(self.heart_check, axis=1)
        fh["sibling_heart"] = fh[["Illnesses of siblings | Instance 0"]].apply(self.heart_check, axis=1)

        fh["father_parkinson"] = fh[["Illnesses of father | Instance 0"]].apply(self.parkinson_check, axis=1)
        fh["mother_parkinson"] = fh[["Illnesses of mother | Instance 0"]].apply(self.parkinson_check, axis=1)
        fh["sibling_parkinson"] = fh[["Illnesses of siblings | Instance 0"]].apply(self.parkinson_check, axis=1)

        fh = fh[['Participant ID',"father_stroke",
                "mother_stroke", "sibling_stroke",
                "father_alzheimer", "mother_alzheimer",
                "sibling_alzheimer", "father_bloodpressure",
                "mother_bloodpressure", "sibling_bloodpressure",
                "father_heart", "mother_heart",
                "sibling_heart","father_parkinson",
                "mother_parkinson","sibling_parkinson" ]]
        return fh

    def pre_physical_activity(self):
        '''
        Returns selected physical activity measures
        '''
        pa = pd.read_csv(self.physical_activity, low_memory=False)
        pa = pa[["Summed MET minutes per week for all activity | Instance 0","Summed minutes activity | Instance 0", "Participant ID" ]]
        return pa
    
    def pre_sleep(self):
        '''
        Returns selected sleep measures
        '''
        sleep = pd.read_csv(self.sleep, low_memory=False)
        sleep = sleep[["Participant ID", "Sleep duration | Instance 0"]]
        sleep = sleep.replace("Prefer not to answer", pd.np.nan)
        sleep = sleep.replace("Do not know", pd.np.nan)
        return sleep
    
    def pre_smoking(self):
        '''
        Returns dataframe with columns indicating if a participant smokes, or has smoked
        '''
        smoking = pd.read_csv(self.smoking, low_memory=False)
        smoking = smoking[["Participant ID","Tobacco smoking"]]
        smoking = pd.get_dummies(smoking, columns=["Tobacco smoking", ])
        smoking = smoking[['Participant ID', 'Tobacco smoking_Ex-smoker',
                            'Tobacco smoking_Never smoked', 'Tobacco smoking_Occasionally',
                            'Tobacco smoking_Smokes on most or all days']]
        return smoking
    
    def pre_white_bloodcell(self):
        '''Returns dataframe with whitebloodcell data
        '''
        wb = pd.read_csv(self.white_bloodcell)
        first_and_general_instances = [i for i in list(wb.columns) if not re.search("Instance 1|Instance 2|Instance 3", i)]
        remove_columns = [ i for i in wb.columns if i not in first_and_general_instances ]
        wb = wb.drop(remove_columns, axis=1)
        return wb

    def earliest_date(self, l):
        '''This function searches for the earliest data in a dataframe row

        Parameters
        ----------
        l : array
            List of row content as used in pd.apply function 
        '''
        l = l.dropna()
        if len(l) == 0:
            return pd.np.nan
        elif len(l) == 1:
            l = l[0].split("-")
            try:
                l = datetime.datetime(int(l[0]), int(l[1]), int(l[2]))
                return l
            except:
                print("Varible not convertable to datetime: ", l)
        else:
            t = []
            for i in l:
                i = i.split("-")
                try: 
                    i = datetime.datetime(int(i[0]), int(i[1]), int(i[2]))
                    t.append(i)
                except:
                    print("Varible not convertable to datetime: ", l)
            return min(t)
        
    def nan_to_binary(self,i):
        '''This funtion converts the exitance of an variable into binary

        Parameters
        ----------
        i : any variable
        '''
        if pd.isnull(i):
            return 0
        return 1
    
    def nan_to_tertiary(self, l):
        '''Converts missing data to zero, and either 1 or 2 if 
        '''
        if pd.isnull(l[1]):
            return 0
        if pd.Timestamp(l[1]) < pd.Timestamp(l[2]):
            return 1
        return 2
    
    def merge_noneoftheabove(self, df):
        return df.replace(to_replace="None of the above (group 1)|None of the above (group 2)", value="None of the above")
    
    def remove_redundant_noneoftheabove(self, df):
        df =  df.replace(to_replace="\|None of the above \(group 1\)", value="", regex=True)
        df =  df.replace(to_replace="\|None of the above \(group 2\)", value="", regex=True)
        return df
    
    def diagnosis_including_diabetes(self, df, df_columns):
        #Consruct three lists:
        # 1) dm: The various sets of diseases containing dm
        # 2) other: The various sets of diseases not containing dm
        # 1) uncertain: The condition of familymember is not specified
        all = []
        for col in df_columns:
            for i in df[col].unique():
                if i not in all:
                    all.append(i)
                    
        dm = []
        other = []
        uncertain = []
        for i in all:
            if type(i) == str:
                if re.search("Diabetes", i):
                    dm.append(i)
                elif re.search("Do not know", i):
                    uncertain.append(i)
                else:
                    other.append(i)
            else:
                uncertain.append(i)

        return dm, other, uncertain
    
    def ethnicity_regrouping(self, df, col_name):  # demographics
  
        try:
                if "Ethnic background" not in col_name:
                    raise ValueError("Double check that the column is the ethnicity background column" )
            
        except ValueError as e:
            print("Error:", e)
            # You can choose to re-raise the exception to propagate it to the calling code
            raise e    
        else:
            ethnicity_mapping = {    # We use the same groups as explained in the AMS.
                'British': 'White',          
                'Irish': 'White',
                'White' : 'White',
                'Any other white background': 'White',
                        
                'Mixed': 'Mixed',
                'White and Black Caribbean': 'Mixed',
                'White and Black African': 'Mixed',
                'White and Asian': 'Mixed',
                'Any other mixed background': 'Mixed',    
                
                'Asian or Asian British': 'Asian',
                'Indian' : 'Asian',
                'Pakistani' : 'Asian',
                'Bangladeshi':  'Asian',
                'Any other Asian background': 'Asian',
                
                'Black or Black British' : 'Black',
                'African' : 'Black',
                'Caribbean': 'Black',
                'Any other Black background': 'Black',  
                
                'Chinese': 'Chinese',
                'Other ethnic group': 'Other'
                                
            }

            # Replace the values in the 'ethnicity' column using the mapping
            df[col_name] = df[col_name].map(ethnicity_mapping)

            return df
        
    def sex_to_binary(self, i):
        if  re.search("Female",str(i)):
            return 0
        if re.search("Male", str(i)):
            return 1
        else:
            return i
        
    def cholesterol_check(self, i):
        if re.search("Cholesterol lowering medication", str(i[0])) or re.search("Cholesterol lowering medication", str(i[1])):
            return 1
        return 0

    def insulin_check(self, i):
        if re.search("Insulin", str(i[0])) or re.search("Insulin", str(i[1])):
            return 1
        return 0

    def Blood_pressure_check(self, i):
        if re.search("Blood pressure medication", str(i[0])) or re.search("Blood pressure medication", str(i[1])):
            return 1
        return 0
    
    def stroke_check(self, i):
        if re.search("Stroke", str(i)):
            return 1
        if re.search("nan", str(i)):
            return pd.np.nan
        return 0

    def alzheimer_check(self, i):
        if re.search("Alzheimer's disease/dementia", str(i)):
            return 1
        if re.search("NaN", str(i)) or re.search('Prefer not to answer', str(i)):
            return pd.np.nan
        else:
            return 0

    def bloodpressure_check(self, i):
        if re.search("High blood pressure", str(i)):
            return 1
        if re.search("nan", str(i)):
            return pd.np.nan
        return 0

    def heart_check(self, i):
        if re.search("Heart disease", str(i)):
            return 1
        if re.search("nan", str(i)):
            return pd.np.nan
        return 0

    def parkinson_check(self, i):
        if re.search("Parkinson's disease", str(i)):
            return 1
        if re.search("nan", str(i)):
            return pd.np.nan
        return 0
    
    def gfr(self, l): #add pre_dem_basic to get correct calculation
        gfr = 175 * (l[3]**-1.154) * (l[26] ** -0.203) 
        if l[27] == 0:
            gfr = gfr*0.742
        if l[29] == 1:
            gfr = gfr*1.212
        return gfr
    
    def preprocessing_factory(self):
        df = self.pre_demographics_basic()
        df = df.merge(self.pre_demographics_ethnicity() ,on="Participant ID")
        df = self.pre_blood_biomarker(dem_basic=df) 
        df = df.merge(self.pre_alcohol() ,on="Participant ID")
        df = df.merge(self.pre_bodymeasures() ,on="Participant ID")
        df = df.merge(self.pre_blood_pressure() ,on="Participant ID")
        df = df.merge(self.pre_family_history() ,on="Participant ID")
        df = df.merge(self.pre_family_history_other() ,on="Participant ID")
        df = df.merge(self.pre_medications() ,on="Participant ID")
        df = df.merge(self.pre_sleep() ,on="Participant ID")
        df = df.merge(self.pre_smoking() ,on="Participant ID")
        df = df.merge(self.pre_urine_biomarkers() ,on="Participant ID")
        df = df.merge(self.pre_physical_activity() ,on="Participant ID")
        df = df.merge(self.pre_white_bloodcell() ,on="Participant ID")
        return df
