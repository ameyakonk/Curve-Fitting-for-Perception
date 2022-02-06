import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils

video1_list = []
x_mat = np.zeros((54,3))
y_mat = np.zeros((54,1))
theta = np.zeros((3,1))
val1 = np.zeros((3,3))

def video1_main():
	vidcap = cv2.VideoCapture('ball_video1.mp4')
	success,image = vidcap.read()
	
	count = 0
	while success:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_red = np.array([155,25,0])
                upper_red = np.array([179,255,255])
                mask = cv2.inRange(hsv, lower_red, upper_red)
                #cv2.imwrite("frame%d.jpg" % count, mask)
                upper_x = 1600;
                upper_y = 1600;
                lower_x = 0;
                lower_y = 0;
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                video1_list.append(extTop)
                video1_list.append(extBot)
                print(extTop)
                print(extBot)
                success,image = vidcap.read()
                count = count+1

def SLS(video1_list):
        for i in range(54):
                x_mat[i,0] = 1
                x_mat[i,1] = video1_list[i][0]
                x_mat[i,2] = video1_list[i][0] * video1_list[i][0]
                y_mat[i,0] = video1_list[i][1]

        val1 = x_mat.transpose().dot(x_mat)
        inverse = np.linalg.inv(val1)
        val2 =  x_mat.transpose().dot(y_mat)
        theta = inverse.dot(val2)

        x = np.linspace(60,2314,50)
        ans = theta[0,0] + theta[1,0]*x + theta[2,0]*x**2
        fig = plt.figure(figsize = (12, 7))
        plt.plot(x, ans, 'r')
        plt.show()
        print(theta)


video1_main()
SLS(video1_list)
              
