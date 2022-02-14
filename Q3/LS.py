import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils
import csv
import random

class LineFitter:
    def __init__(self):
        file = open('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
        type(file)
        csvreader = csv.reader(file)
        self.x_mat = []
        self.y_mat = []
        rows = []
        for row in csvreader:
                rows.append(row)
        for i in range(1,len(rows)):
            self.x_mat.append(float(rows[i][0]))
            self.y_mat.append(float(rows[i][6]))

        self.min_x = min(self.x_mat)
        self.min_y = min(self.y_mat)
        self.max_x = max(self.x_mat)
        self.max_y = max(self.y_mat)
        
        x_mat = self.normalise(self.x_mat)
        y_mat = self.normalise(self.y_mat)

        print("Processing.....")
        
    def getData(self):
        return self.x_mat, self.y_mat

    def getAvg(self, mat):
        return sum(mat)/len(mat)

    def normalise(self, mat):
        for i in range(len(mat)):
            mat[i] = (mat[i] - min(mat))/(max(mat) - min(mat))
        return mat

    def denormalise(self, mat, min_, max_):
        for i in range(len(mat)):
            mat[i] =  mat[i]*(max_ - min_) + min_ 
        return mat
    
    def leastSquare(self, x_mat, y_mat):
        X = np.ones((len(x_mat),2))
        Y = np.zeros((len(x_mat),1))
        xPoints = []
        yPoints = []
        for i in range(len(x_mat)):
            X[i,1] = x_mat[i]
            Y[i,0] = y_mat[i]
            xPoints.append(x_mat[i])
            yPoints.append(y_mat[i])

        Theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))

        x = np.linspace(0.3,1,50)
        y = Theta[0,0] + Theta[1,0]*x
        
        y = self.denormalise(y, self.min_y, self.max_y)
        x = self.denormalise(x, self.min_x, self.max_x)
        
        xPoints = self.denormalise(xPoints, self.min_x, self.max_x)
        yPoints = self.denormalise(yPoints, self.min_y, self.max_y)

        return x,y,xPoints,yPoints,Theta

    def totalleastSquare(self):
        x_mat, y_mat = self.getData()
        x_mat = self.normalise(x_mat)
        y_mat = self.normalise(y_mat)
        x_mat_avg = self.getAvg(x_mat)
        y_mat_avg = self.getAvg(y_mat)
        U = np.zeros((len(x_mat), 2))
        for i in range(len(x_mat)):
            U[i,0] = x_mat[i] - x_mat_avg
            U[i,1] = y_mat[i] - y_mat_avg

        w, v = LA.eig(U.transpose().dot(U))
        min_eigen = 1 if w[0] >= w[1] else 0
        eigen_solution = np.zeros((2,1))

        eigen_solution[0,0] = v[0, min_eigen]     # a
        eigen_solution[1,0] = v[1, min_eigen]     # b
        d = eigen_solution[0,0]*x_mat_avg + eigen_solution[1,0]*y_mat_avg   # d

        Theta = np.zeros((2,1))     # Theta = [d/b; a/b]
        Theta[0,0] = d/eigen_solution[1,0]  # Theta[0,0] = d/b
        Theta[1,0] = eigen_solution[0,0]/eigen_solution[1,0] # Theta[1,0] = a/b

        x = np.linspace(0.3,1,50)
        y = Theta[0,0] - Theta[1,0]*x
        
        y = self.denormalise(y, self.min_y, self.max_y)
        x = self.denormalise(x, self.min_x, self.max_x)
        
        return x, y

    def costFunction(self, x, y, Theta):
        difference = 0
        cost = 0
        for i in range(len(x)):
            h_theta = Theta[0,0] + Theta[1,0]*x[i]
            difference =  (h_theta - y[i])**2
            cost = cost + difference
        return cost/(2*len(x));
        
    def RANSAC(self):
        max_Iterations = 1200
        ransac_minpoint = 20 #5
        ransac_threshold = 0.8 #5000
        ransac_miniter = 10
        x_mat, y_mat = self.getData()
        bestErr = 100000000
        FinalTheta = np.zeros((2,1))
        random_list = list(zip(x_mat, y_mat))
        
        for i in range(max_Iterations):
            random.shuffle(random_list)
            unzipped_object = zip(*random_list)
            temp = list(unzipped_object)
            random_x = list(temp[0])
            random_y = list(temp[1])
            
            x,y,xPoints,yPoints,Theta = self.leastSquare(random_x[:ransac_minpoint], random_y[:ransac_minpoint])
            also_inliers_x = []
            also_inliers_y = []
            thisErr = 0
            for i in range(len(random_x[ransac_minpoint:])):
                x_ = random_x[ransac_minpoint:][i]
                y_ = random_y[ransac_minpoint:][i]
                h_theta = Theta[0,0] + Theta[1,0]*x_
                if (y_ - h_theta)**2 < ransac_threshold:
                    also_inliers_x.append(x_)
                    also_inliers_y.append(y_)
            
            also_inliers_x = also_inliers_x + random_x[ransac_minpoint:]
            also_inliers_y = also_inliers_y + random_y[ransac_minpoint:]
            
            if len(also_inliers_x) > ransac_minpoint:
                x,y,xPoints,yPoints,Theta_updated = self.leastSquare(also_inliers_x, also_inliers_y)
                thisErr = self.costFunction(also_inliers_x, also_inliers_y, Theta_updated)
                if thisErr < bestErr:
                    bestErr = thisErr
                    FinalTheta = np.array(Theta_updated)
        x = np.linspace(0.3,1,50)
        y = FinalTheta[0] + FinalTheta[1]*x

        y = self.denormalise(y, self.min_y, self.max_y)
        x = self.denormalise(x, self.min_x, self.max_x)
        
        return x, y
    
    def RANSAC_(self):
        max_Iterations = 50
        ransac_minpoint = 10 #5
        ransac_threshold = 0.03 #5000
        d = 0
        x_mat, y_mat = self.getData()
        FinalTheta = np.zeros((2,1))
        random_list = list(zip(x_mat, y_mat))
        
        for i in range(max_Iterations):
            random.shuffle(random_list)
            unzipped_object = zip(*random_list)
            temp = list(unzipped_object)
            random_x = list(temp[0])
            random_y = list(temp[1])
            
            x,y,xPoints,yPoints,Theta = self.leastSquare(random_x[:ransac_minpoint], random_y[:ransac_minpoint])
            count = 0
            thisErr = 0
            for i in range(len(random_x[ransac_minpoint:])):
                x_ = random_x[ransac_minpoint:][i]
                y_ = random_y[ransac_minpoint:][i]
                h_theta = Theta[0,0] + Theta[1,0]*x_
                if (h_theta - y_) < ransac_threshold:
                    count = count + 1
   
            if count > d:
                d = count
                FinalTheta = Theta
             
        x = np.linspace(0.3,1,50)
        y = FinalTheta[0] + FinalTheta[1]*x

        y = self.denormalise(y, self.min_y, self.max_y)
        x = self.denormalise(x, self.min_x, self.max_x)
        
        return x, y

    def plot(self):
        x_mat, y_mat = self.getData()
        x,y,xPoints,yPoints,_ = self.leastSquare(x_mat, y_mat)
        x_tls, y_tls = self.totalleastSquare()
        x_ransac, y_ransac = self.RANSAC_()
        
        fig = plt.figure(figsize = (12, 7))
        plt.plot(x, y, 'r', label="Least Square")
        plt.plot(x_tls, y_tls, 'g', label="Total Least Square")
        plt.plot(x_ransac, y_ransac, 'k-', label="RANSAC")

        plt.xlabel("age")
        plt.ylabel("charge")
        
        plt.scatter(xPoints, yPoints)
        print("Done!!!")
        plt.legend()
        plt.show()

        
p1 = LineFitter()
p1.plot()
