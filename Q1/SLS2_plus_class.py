import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils



class SLS:
    
    def __init__(self, vidcap):
        self.video1_list = []
        self.theta = np.zeros((3,1))
        self.Row = 0
        self.vidcap = vidcap
        self.count = 0
        #video_main()
    def video_main(self):
            vidcap = self.vidcap
            success,image = vidcap.read()
            print(image.shape)
            row, clm,_ = image.shape
            print(row)
            self.Row = row
            while success:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([155,25,0])
                    upper_red = np.array([179,255,255])
                    mask = cv2.inRange(hsv, lower_red, upper_red)
                    #cv2.imwrite("frame%d.jpg" % count, mask)
                    upper_x = 1676;
                    upper_y = 1676;
                    lower_x = 0;
                    lower_y = 0;
                    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    c = max(cnts, key=cv2.contourArea)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    self.video1_list.append(extTop)
                    self.video1_list.append(extBot)
                    #print(extTop)
                    #print(extBot)
                    success,image = vidcap.read()
                    self.count = self.count+1

    def SLS(self):
            print(self.count)
            x_mat = np.zeros((self.count*2,3))
            y_mat = np.zeros((self.count*2,1))
            for i in range(self.count*2):
                    x_mat[i,0] = 1
                    x_mat[i,1] = self.video1_list[i][0]
                    x_mat[i,2] = self.video1_list[i][0] * self.video1_list[i][0]
                    y_mat[i,0] = self.Row - self.video1_list[i][1]
            #print(Row)
            self.theta = np.linalg.inv(x_mat.transpose().dot(x_mat)).dot(x_mat.transpose().dot(y_mat))
            
            x = np.linspace(60,2314,50)
            ans = self.theta[0,0] + self.theta[1,0]*x + self.theta[2,0]*x**2
            fig = plt.figure(figsize = (12, 7))
            plt.plot(x, ans, 'r')
            plt.show()
        #    print(theta)
            plt.ioff()

p1 = SLS(cv2.VideoCapture('ball_video1.mp4'))
p1.video_main()
p1.SLS()

p2 = SLS(cv2.VideoCapture('ball_video2.mp4'))
p2.video_main()
p2.SLS()        
