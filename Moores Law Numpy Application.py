'''
Determining Moore’s Law with real data in NumPy
What you’ll do
In 1965, engineer Gordon Moore predicted that transistors on a chip would double every two years in the coming decade [1, 2].
You’ll compare Moore’s prediction against actual transistor counts in the 53 years following his prediction.
You will determine the best-fit constants to describe the exponential growth of transistors on semiconductors compared to Moore’s Law.
'''
# Skills you’ll learn
    # Load data from a *.csv file

    # Perform linear regression and predict exponential growth using ordinary least squares

    # You’ll compare exponential growth constants between models

    # Share your analysis in a file:

    # as NumPy zipped files *.npz

    # as a *.csv file

    # Assess the amazing progress semiconductor manufacturers have made in the last five decades
'''
What you’ll need
1. These packages:

NumPy

Matplotlib

imported with the following commands
'''

import matplotlib.pyplot as plt
import numpy as np

'''
You’ll use these NumPy and Matplotlib functions:

    np.loadtxt: this function loads text into a NumPy array
    np.log: this function takes the natural log of all elements in a NumPy array
    np.exp: this function takes the exponential of all elements in a NumPy array
    lambda: this is a minimal function definition for creating a function model
    plt.semilogy: this function will plot x-y data onto a figure with a linear x-axis and 
    y-axis plt.plot: this function will plot x-y data on linear axes
    slicing arrays: view parts of the data loaded into the workspace, slice the arrays e.g. x[:10] for the first 10 values in the array, x    
    boolean array indexing: to view parts of the data that match a given condition use boolean operations to index an array    
    np.block: to combine arrays into 2D arrays    
    np.newaxis: to change a 1D vector to a row or column vector
    np.savez and np.savetxt: these two functions will save your arrays in zipped array format and text, respectively
    
'''
