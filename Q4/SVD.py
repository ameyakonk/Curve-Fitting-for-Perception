import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils
import csv
import random
import math

class SVD:
    def sortArr(self, mat, values):
        dict_ = {}
        for i in range(len(mat)):
            dict_[values[i]] = mat[:,i]
        values = np.sort(values)
        n_mat = np.zeros((len(mat),len(mat)))
        for i in range (len(mat)):
            n_mat[:, i] = dict_[values[i]]
        return n_mat
        #values = np.sort(values)
        
    def svd(self):
        x = [0, 5, 150, 150, 5]
        y = [0, 5, 5, 150, 150]
        xp = [0, 100, 200, 220, 100]
        yp = [0, 100, 80, 80, 200]
        A = np.array([[-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                 [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                 [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                 [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
                 [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                 [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]],
                 [-x[4], -y[4], -1, 0, 0, 0, x[4]*xp[4], y[4]*xp[4], xp[4]],
                 [0, 0, 0, -x[4], -y[4], -1, x[4]*yp[4], y[4]*yp[4], yp[4]]])

        eig1, U = LA.eig(A.dot(A.transpose()))
        eig2, V = LA.eig(A.transpose().dot(A))
        #print(eig1)
        #print(eig2)
        #U = self.sortArr(U, eig1)
        #V = self.sortArr(V, eig2)
        
        eig1 = eig1**0.5
        eig1 = np.sort(eig1)[::-1]
        lambda_ = np.zeros((8,9))
        val = 0
        for i in range(8):
            lambda_[i, i] = eig1[i]
        #ans = (U.dot(lambda_)).dot(V.transpose())
        U_, S_, V_ = np.linalg.svd(A)
        ans = (U_.dot(lambda_)).dot(V_)
        
        print(V.T)
        print(" ")
        print(V_)
        
        #ans = (V.dot(lambda_)).dot(U.transpose())
        #print(ans)
        #print(A)
p1 = SVD()
p1.svd()
