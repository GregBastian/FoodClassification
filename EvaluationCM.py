# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:27:02 2019

@author: Gregorius Ivan Sebastian
@eMail : greg.bastian@student.ub.ac.id
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:05:42 2019

@author: User
"""

from ImageCM import Image
import csv
import os
import ast

def getTrainingData(training_wd):
    '''
    training_wd : the directory where the CSV file containing the preprocessed
                    file is stored (check DataProcessing.py)
    returns all of the processed image files containing their 
                    respective features from a csv file
    '''    
    os.chdir(training_wd)
    path, dirs, files = next(os.walk(training_wd))
    file_count = len(files) - 1
    i = 0
    trainingImages = []
    
    with open('TRAINING DATABASE.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            filename = row[0]
            print('Processing...', filename)
            food_name = row[1]
            cm = ast.literal_eval(row[3])
            preloadedData = [food_name,filename,cm]
            trainingEntry = Image(data = preloadedData, preload_status = True)
            trainingImages.append(trainingEntry)
            i += 1
            percentage = str(round(i/file_count * 100, 2))
            print('Progress of training... '+ percentage +'% Done')
            
    return trainingImages


def getTestingData(testing_wd):
    '''
    testing_wd : the directory where the CSV file containing the preprocessed
                    file is stored (check DataProcessing.py)
    returns all of the processed image files containing their 
                    respective features from a csv file
    '''    
    os.chdir(testing_wd)
    path, dirs, files = next(os.walk(testing_wd))
    file_count = len(files) - 1
    i = 0
    testingImages = []
    
    with open('TESTING DATABASE.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            filename = row[0]
            print('Processing...', filename)
            food_name = row[1]
            cm = ast.literal_eval(row[3])
            preloadedData = [food_name,filename,cm]
            testingEntry = Image(data = preloadedData, preload_status = True)
            testingImages.append(testingEntry)
            i += 1
            percentage = str(round(i/file_count * 100, 2))
            print('Progress of testing... '+ percentage +'% Done')
            
    return testingImages


def normalizeTrainingData(trainingImages):
    '''
    trainingImages : a list containing all preprocessed testing data
    returns all of the processed testing files after being normalized
    '''    
    for trainingEntry in trainingImages:
        trainingEntry.set_data_normalized()
    return trainingImages


def normalizeTestingData(testingImages):
    '''
    testingImages : a list containing all preprocessed training data
    returns all of the processed testing files after being normalized
    '''    
    for testingEntry in testingImages:
        testingEntry.set_data_normalized()
    return testingImages


def getEvaluation(trainingImages, testingImages):
    '''
    trainingImages : a list containing all normalized preprocessed training data
    testingImages : a list containing all normalized preprocessed testing data
    returns None but prints the evaluation result on the console
    '''

    def euclideanDistance(item_1,item_2):
        '''
        item 1: numpy array containing features from item_1
        item 2: numpy array containing features from item_2
        returns euclidean distance between position of item_1 and item_2
        '''
        return (sum((item_1 - item_2)**2))**0.5
    
    def manhattanDistance(item_1, item_2):
        '''
        item_1 : numpy array containing features from item_1
        item_2 : numpy array containing features from item_2
        returns manhattan distance between position of item_1 and item_2
        '''
        return sum(abs(item_1 - item_2))
    
    from sklearn.metrics import classification_report
    true_y = []
    predicted_y = []
    NEIGHBOURS = 9
    for testItem in testingImages:
        true_y.append(testItem.get_food_name())
        pos_testing = testItem.get_data_normalized()
        neighbors = {}
        for training_item in trainingImages:
            pos_training = training_item.get_data_normalized()
            distance = euclideanDistance(pos_testing, pos_training)
            neighbors[distance] = training_item.get_food_name()
            closestNeighbors = [neighbors[key] for key in sorted(neighbors.keys())[:NEIGHBOURS]]
        result = max(closestNeighbors, key=closestNeighbors.count)
        predicted_y.append(result)
    finalResults = classification_report(true_y, predicted_y)
    print(finalResults)


if __name__ == "__main__":
    
    training_wd = r'G:\My Drive\Skripsi Jaya!\DATASET - Original\ALL Training Data' 
    testing_wd = r'G:\My Drive\Skripsi Jaya!\DATASET - Original\ALL Testing Data'  
    trainingImages = getTrainingData(training_wd) 
    testingImages = getTestingData(testing_wd)
    trainingImages = normalizeTrainingData(trainingImages)
    testingImages = normalizeTestingData(testingImages)
    getEvaluation(trainingImages, testingImages)
    



    