import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils

class SLS:
    
    def __init__(self):
        self.Row = 0
        print("Processing....")
     
    def video_main(self, vidcap):
            video1_list = []
            success,image = vidcap.read()
            row, clm,_ = image.shape
            self.Row = row
            count = 0
            while success:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([155,25,0])
                    upper_red = np.array([179,255,255])
                    mask = cv2.inRange(hsv, lower_red, upper_red)
                    upper_x = 1676;
                    upper_y = 1676;
                    lower_x = 0;
                    lower_y = 0;
                    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    c = max(cnts, key=cv2.contourArea)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    video1_list.append(extTop)
                    video1_list.append(extBot)
                    success,image = vidcap.read()
                    count += 1
            return self.SLS(video1_list, count)
                
    def SLS(self, video1_list, count):
            x_mat = np.zeros((count*2,3))
            y_mat = np.zeros((count*2,1))
            xPoints = []
            yPoints = []
            theta = np.zeros((3,1))
        
            for i in range(count*2):
                    
                    x_mat[i,0] = 1
                    x_mat[i,1] = video1_list[i][0]
                    x_mat[i,2] = video1_list[i][0] * video1_list[i][0]
                    y_mat[i,0] = self.Row - video1_list[i][1]
                    xPoints.append(video1_list[i][0])
                    yPoints.append(video1_list[i][1])

            theta = np.linalg.inv(x_mat.transpose().dot(x_mat)).dot(x_mat.transpose().dot(y_mat))
            return theta, xPoints, yPoints, x_mat, y_mat            

    def plot_(self):
            theta1, xPoints1, yPoints1, x_mat1, y_mat1 = self.video_main(cv2.VideoCapture('ball_video1.mp4'))
            x1 = np.linspace(40,2340,50)
            ans1 = theta1[0,0] + theta1[1,0]*x1 + theta1[2,0]*x1**2

            theta2, xPoints2, yPoints2, x_mat2, y_mat2 = self.video_main(cv2.VideoCapture('ball_video2.mp4'))
            x2 = np.linspace(40,3440,50)
            ans2 = theta2[0,0] + theta2[1,0]*x2 + theta2[2,0]*x2**2
            
            fig, axs = plt.subplots(2)
            axs[0].plot(x1, ans1, 'r', label = "Video1 Least Square")
            axs[1].plot(x2, ans2, 'g', label = "Video2 Least Square")
            axs[0].scatter(xPoints1, max(yPoints1) - yPoints1, label = "Video1 ball coordinates")
            axs[1].scatter(xPoints2, max(yPoints2) - yPoints2, label = "Video2 ball coordinates")
            axs[0].set_ylabel('Y-axis')

            axs[1].set_xlabel('X-axis')
            axs[1].set_ylabel('Y-axis')
           
            axs[0].legend(loc='upper right', prop={"size":5})
            axs[1].legend(loc='upper right', prop={"size":5})
            
            print("Done!!!")
            plt.show()
          
p1 = SLS()
p1.plot_()
        
