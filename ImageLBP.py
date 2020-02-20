# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:45:20 2019

@author: Gregorius Ivan Sebastian
@eMail : greg.bastian@student.ub.ac.id
"""

import cv2
import numpy as np

class Image:
    
    max_value = min_value = None
    #for data normalization
    
    def __init__(self, data = [], preload_status = False):
        #preload status if false means files are not from preloaded source
        #preloaded source to quicken training time
        bin_size = 8
        
        if preload_status == False:
            preprocessed_image = Image.preprocessing(data[1])
            resized_image = Image.resize(preprocessed_image, 450, 450)
            self.food_name = data[0].split('_')[1]
            self.file_name = data[0]
            self.LBP = Image.set_LBP(resized_image)
            self.CM = Image.set_colorMoments(resized_image)
            self.set_data(bin_size)
            self.data_normalized = None
        else:
            self.food_name = data[0]
            self.file_name = data[1]
            self.LBP = data[2]
            self.set_data(bin_size)
            self.data_normalized = None
            
        
    def preprocessing(img): 
        def grayscalling(img):
            '''
            img : a colored image in RGB
            returns an image in grayscale
            '''
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return result
        
        def gaussian_thresh(img):
            '''
            img : an image grayscale
            return an image with erosion and dilation with 5 iterations respectively using a
                5x5 (0) matrix
            '''
            img = cv2.GaussianBlur(img,(5,5),0)
            th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) 
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 15)
            erosion = cv2.dilate(opening,kernel, iterations = 3)
            result = cv2.bitwise_not(erosion)
            return result  
        
        grayscale = grayscalling(img)
        thresholded = cv2.cvtColor(gaussian_thresh(grayscale), cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(img, thresholded)
        
    
    def resize (img, x, y):
        '''
        x,y : dimensions of resizing
        img : src image that wants to be resized
        returns a resized image with dimensions x and y
        '''
        resize = (x, y)
        resized = cv2.resize(img, resize)
        return resized
        
    def set_data(self,bin_size):
        '''
        self : the object. This method must be called thru an instance of an object
        bin_size : to deternime how many bins will be used to be used in LBP
        '''
        
        def data_LBP(bin_size):
            '''
            bin_size : specifiy to group LBP values based on bin size
            return an array with total frequency respective to bin size
            '''
            lbp_array = []
            for bin_range in range(0,256,bin_size):
                start = bin_range
                end = start + bin_size
                total_frequencies =  +\
                sum([self.LBP[key] for key in range(start,end)])
                lbp_array.append(total_frequencies)
            return np.array(lbp_array)
        
        
        # concatanate LBP and color moments data into 1 dimensional matrix  
        self.data =  data_LBP(bin_size)
        # keeping value for normalization
        if Image.max_value == None and Image.min_value == None:
            Image.max_value = max(self.data)
            Image.min_value = min(self.data)
            print('-> Min and Max Value updated!')
        else:
            if Image.max_value < max(self.data):
                Image.max_value = max(self.data)
                print('-> Maximum Value updated!')
            if Image.min_value > min(self.data):
                Image.min_value = min(self.data)
                print('-> Mininum Value updated!')
                
    def set_data_normalized(self):
        '''
        self: the object that this method is called
        returns the normalized data of said object in a one dimensional array
            the normalizatin
        '''
        self.data_normalized = + \
        (self.data - Image.min_value) / (Image.max_value - Image.min_value)
        
    def get_LBP(self):
        '''
        self: the object that this method is called
        returns a dictionary containing the object's local binary patterns
        '''
        return self.LBP.copy()  
        
    def get_data(self):
        '''
        self: the object that this method is called
        returns the data of said object in one dimensional numpy array
        '''
        return self.data
    
    def get_data_normalized(self):
        '''
        self: the object that this method is called
        returns the NORMALIZED data of said object in one dimensional numpy array
        '''
        return self.data_normalized
    
    def get_food_name(self):
        '''
        self: the object that this method is called
        returns the food name of object in a string
        '''
        return self.food_name
    
    def get_file_name(self):
        '''
        self: the object that this method is called
        returns the ffilename (complete with extension) of object in a string
        '''
        return self.file_name