# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:54:47 2019

@author: Gregorius Ivan Sebastian
@eMail : greg.bastian@student.ub.ac.id
"""

from ImageLBP_and_CM import Image
import csv
import os
import cv2

def processTrainingData(training_wd):
    '''
    training_wd : the directory where all of the training data
                    (in jpeg format) is stored
    returns all of the processed image files containing their 
                    respective features
    '''
    os.chdir(training_wd)
    path, dirs, files = next(os.walk(training_wd))
    totalFiles = len(files) - 1
    totalFilesProcessed = 0
    trainingImages = []
        
    for filename in os.listdir(training_wd):
        if filename.endswith('jpeg'):
            print('Processing...', filename)
            currentImage = cv2.imread(filename)
            trainingEntry = Image(data = [filename, currentImage])
            trainingImages.append(trainingEntry)
            print(filename, 'has finished processing!')
            totalFilesProcessed += 1
            percentage = str(round(totalFilesProcessed/totalFiles * 100, 2))
            print('Progress of training... '+percentage+'% Done')
            print('===========================================')
    return trainingImages.copy()
         
       
def processTestingData(testing_wd):
    '''
    testing_wd : the directory where all of the testing data
                    (in jpeg format) is stored
    returns all of the processed image files containing their 
                    respective features
    '''      
    os.chdir(testing_wd)
    path, dirs, files = next(os.walk(testing_wd))
    totalFiles = len(files) - 1
    totalFilesProcessed = 0
    testingImages = []
    
    for filename in os.listdir(testing_wd):
        if filename.endswith('jpeg'):
            print('Processing...', filename)
            currentImage = cv2.imread(filename)
            testingEntry = Image(data = [filename, currentImage])
            testingImages.append(testingEntry)
            print(filename, 'has finished processing!')
            totalFilesProcessed += 1
            percentage = str(round(totalFilesProcessed/totalFiles * 100, 2))
            print('Progress of testing... '+percentage+'% Done')
            print('===========================================')
    return testingImages.copy()      
      


               
def saveTrainingData(training_wd, training_images):
    '''
    training_wd : the directory where you want to save the processed data
                    for the images
    training_images : a list with all objects containing data of all processed
                    images
    returns None, but creates a csv file named 'TRAINING DATABASE' 
                    in directory contained in training_wd
    '''
    os.chdir(training_wd)
    with open('TRAINING DATABASE.csv', mode='w', newline = '') as test_file:
        for training_entry in training_images:
            image_writer = csv.writer(test_file, delimiter=',', quotechar='"', 
                                      quoting=csv.QUOTE_MINIMAL)
            file_name = training_entry.get_file_name()
            food_name = training_entry.get_food_name()
            lbp = training_entry.get_LBP()
            cm = training_entry.get_CM()
            image_writer.writerow([file_name, food_name, lbp, cm])
            
    totalDataSaved = len(training_images)
    print('There were', totalDataSaved , 'training data saved!')
   
    
    

def saveTestingData(testing_wd, testing_images):
    '''
    testing_wd : the directory where you want to save the processed data
                    for the images
    testing_images : a list with all objects containing data of all processed
                    images
    returns None, but creates a csv file named 'TESTING DATABASE' 
                    in directory contained in testing_wd
    '''             
    os.chdir(testing_wd)
    with open('TESTING DATABASE.csv', mode='w', newline = '') as test_file:
        for testingEntry in training_images:
            image_writer = csv.writer(test_file, delimiter=',', quotechar='"', 
                                      quoting=csv.QUOTE_MINIMAL)
            file_name = testingEntry.get_file_name()
            food_name = testingEntry.get_food_name()
            lbp = testingEntry.get_LBP()
            cm = testingEntry.get_CM()
            image_writer.writerow([file_name, food_name, lbp, cm])
            
    totalDataSaved = len(testing_images)
    print('There were', totalDataSaved , 'testing data saved!')
                 


if __name__ == "__main__":
    
    training_wd = r'G:\My Drive\Skripsi Jaya!\DATASET - Original\ALL Training Data' 
    testing_wd = r'G:\My Drive\Skripsi Jaya!\DATASET - Original\ALL Testing Data'  
    training_images = processTrainingData(training_wd) 
    testing_images = processTestingData(testing_wd)
    saveTrainingData(training_images)
    saveTestingData(testing_images)

