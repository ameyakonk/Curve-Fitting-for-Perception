# Curve-Fitting-for-Perception

## Overview
The repository contains the following contents.

1. Nonlinear Curve Fitting of a ball trajectory using Standard Least Square method.
2. Covariance matrix calculation of a age vs charge dataset.
3. Linear Curving Fitting of the age vs charge dataset using Least Square, Total Least Square and RANSAC.
4. Calculating SVD of a matrix and computing Homography matrix for the same. 

## Personnel
### Ameya Konkar 

UID:118191058

Master's Student at University of Maryland, College Park

## Dependencies 

Install the following dependencies:

1.  Install Anaconda Navigator to create multiple environments for future
    projects. Anaconda navigator also gives you access to Spyder and Jupyter
    Notebook, very useful tools for python coding. For windows, refer here. For
    Linux, refer [here](https://docs.anaconda.com/anaconda/install/linux/).
2.  Install required packages onto your virtual environment. Replace “myenv”
    with your environment name. Enter the following commands in your
    terminal window. Press ‘y’ when prompted. (Step h just launches
    the spyder application). Remember to always work in your virtual
    environment to properly run your codes .
    a. conda create -n myenv python=3.7
    b. conda activate myenv
    c. conda install -c conda-forge opencv=4.1.0
    d. conda install -c anaconda numpy
    e. conda install -c conda-forge matplotlib
    f. conda install -c conda-forge imutils
 
### Building the Program and Tests

```
sudo apt-get install git
git clone --recursive https://github.com/ameyakonk/Curve-Fitting-for-Perception.git
cd <path to repository>
conda activate <name of env>
```

Q2. To Run the code:
```
cd Q2/
chmod +x CurveFitter.py
python CurveFitter.py 
```

Q3. To Run the code:  
```
cd Q3/
 1. To run the program that computes covariance matrix
    chmod +x Covariance.py
    python Covariance.py
 
 2. To run the program that computes Least Square, Total Least Square, RANSAC
    chmod +x LS.py
    python LS.py
```   
Q4. To Run the code:
```
cd Q4/
chmod +x SVD.py
python SVD.py 
```
