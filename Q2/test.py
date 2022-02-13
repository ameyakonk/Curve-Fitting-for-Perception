import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imutils


def main():
    image = cv2.imread('/home/ameya/UMD/ENPM-673/Assignment-1/frame1.jpg',0)
    rows,cols = image.shape
    upper_x = 1600;
    upper_y = 1600;
    lower_x = 0;
    lower_y = 0;
    for i in range(rows):
        for j in range(cols):
            k = image[i,j]
            if k != 255:
                if i > lower_x:
                    lower_x = i
                    lower_y = j
                
                if i < upper_x:
                    upper_x = i
                    upper_y = j

    print(lower_x, end=" ")
    print(lower_y, end=" ")
    print(upper_x, end=" ")
    print(upper_y)
    
    cv2.destroyAllWindows()

main()