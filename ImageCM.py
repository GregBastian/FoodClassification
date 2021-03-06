# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:27:19 2019

@author: Gregorius Ivan Sebastian
@eMail : greg.bastian@student.ub.ac.id
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:46:59 2018

@author: User
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
            self.CM = data[2]
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
    

    def set_colorMoments(img):
        '''
        img : a segmented image with RGB color channel
        returns a 3x3 matrix where the rows are the mean , standard deviation 
                and skewness and the columns represent color channel B,G,R in order
                from top to bottom
        '''
        def get_mean(channel):
            N = channel.size
            mean = 1/N * np.sum(channel)
            return mean
        
        def get_std(channel, mean):
            N = channel.size
            std = (1/N * np.sum((channel-mean)**2)) ** (1/2)
            return std
        
        def get_skewness(channel, mean):
            N = channel.size
            skewness = (1/N * abs(np.sum((channel-mean)**3))) ** (1./3)
            # an abs() method is used just in case '(channel-mean)**3' results
            # in a negative number, because finding the cubic root of a negative 
            # number results in an error
            return skewness
        
        result = []
        for index in range(3):
            channel = img[:,:,index]
            mean = get_mean(channel)
            sd = get_std(channel, mean)
            skewness = get_skewness(channel, mean)
            result += [mean, sd, skewness]
        return result


    def set_data(self,bin_size):
        '''
        self : the object. This method must be called thru an instance of an object
        bin_size : to deternime how many bins will be used to be used in LBP
        '''
        
        def data_CM():
            '''
            return an array which contains color moments for the current image
            '''
            return np.hstack(self.CM) 
        
        # concatanate LBP and color moments data into 1 dimensional matrix  
        self.data = data_CM()
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
        

    
    def get_CM(self):
        '''
        self: the object that this method is called
        returns a numpy array containing the object's color moments
        '''
        return self.CM.copy()
        
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
    
            
            
            
                