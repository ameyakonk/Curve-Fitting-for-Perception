import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils
import csv


class Covariance:
        
    def readFile(self):
        print("Processing...")
        file = open('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
        type(file)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        x_mat = []
        y_mat = []
        for i in range(1,len(rows)):
            x_mat.append(float(rows[i][0]))
            y_mat.append(float(rows[i][6]))
            
        self.covarianceMat(x_mat, y_mat)

    def covarianceMat(self, x_mat, y_mat):
        x_mat_avg = self.average(x_mat)
        y_mat_avg = self.average(y_mat)

        x_mat_var = self.variance(x_mat, x_mat_avg)
        y_mat_var = self.variance(y_mat, y_mat_avg)

        covariance = self.covariance(x_mat, x_mat_avg, y_mat, y_mat_avg)
       
        covariance_mat = np.array([[x_mat_var, covariance],[covariance, y_mat_var]])
        print("Covariance Matrix", end=" ")
        print(covariance_mat)
        
        w, v = LA.eig(covariance_mat)
        print("Eigen Values", end=" ")
        print(w)
        print("Eigen Vectors", end=" ")
        print(v)
        self.plot(v, x_mat, y_mat)
        
    def average(self, list_name):
        return sum(list_name)/len(list_name)

    def variance(self, list_name, avg):
        list_name = [(x - avg)**2 for x in list_name]
        return sum(list_name)/len(list_name)
        
    def covariance(self, list1, avg1, list2, avg2):
        temp_1 = [(x - avg1) for x in list1]
        temp_2 = [(x - avg2) for x in list2]
        temp_3 = [temp_1[i]*temp_2[i] for i in range(len(list2))]
        return sum(temp_3)/len(temp_3)

    def plot(self, eigen_vector, x, y):
        xPoints = []
        yPoints = []
        for i in range(len(x)):
            xPoints.append(x[i])
            yPoints.append(y[i])

        origin = [0, 0]
        eig_vec1 = eigen_vector[:,0]
        eig_vec2 = eigen_vector[:,1]
        plt.quiver(*origin, *eig_vec1, color=['r'], scale=21, label = "eigenvector1")
        plt.quiver(*origin, *eig_vec2, color=['b'], scale=21, label = "eigenvector2")
        plt.xlabel('age')
        plt.ylabel('charges')
        plt.scatter(xPoints, yPoints)
        
        plt.legend()
        print("Done...")
        plt.show()
    
instance = Covariance()
instance.readFile()
