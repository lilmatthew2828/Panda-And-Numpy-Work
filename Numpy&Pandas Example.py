"""
To try the examples in the browser:
1. Type code in the input cell and press
   Shift + Enter to execute
2. Or copy paste the code, and click on
   the "Run" button in the toolbar
"""
'''
Top Python Libraries for Data Science:
NumPy
Keras
Pandas
PyTorch
SciPy
Scikit-Learn
TensorFlow
Matplotlib
Seaborn
Theano
'''

# The standard way to import NumPy:
import numpy as np
from tkinter import *
import numpy
from tabulate import tabulate
import random
import pygame
import matplotlib
import pandas
import torch
from sklearn import metrics
from sklearn.utils.extmath import density
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC



'''
df = pandas.DataFrame({"Name": ["Braund, Mr. Owen Harris", "Allen, Mr. William Henry", "Bonnell, Miss. Elizabeth", ],
                       "Age": [22, 35, 58],
                       "Sex": ["male", "male", "female"], })
'''


'''
# Basic Reading Example Code:
a = numpy.arange(6)
a2 = a[numpy.newaxis, :]
print(a2.shape)
'''
'''
What’s the difference between a Python list and a NumPy array?

NumPy gives you an enormous range of fast and efficient ways of creating arrays and manipulating numerical data inside them. 
While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogeneous. 
The mathematical operations that are meant to be performed on arrays would be extremely inefficient if the arrays weren’t homogeneous.

Why use NumPy?

NumPy arrays are faster and more compact than Python lists. 
An array consumes less memory and is convenient to use.
NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. 
This allows the code to be optimized even further.

# What is an Array?8
# An array is a central data structure of the Numpy Library.
# An is array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpet an element.
# It has a grid of elements that can be indexed in various ways.
# The elements are all of the same type, referred to as the array dtype
# An array can be indexed by a tuple of nonnegative integers, by booleans,
# by another array, or by integers. The rank of the array is the number of dimensions.
# The shape of the array is a tuple of integers giving the size of the array along each dimension.
'''
'''
ndarray(shape[, dtype, buffer, offset, ...])
An array object represents a multidimensional, homogeneous array of fixed-size items.
'''
print()
# The number of dimensions and items in an array is defined by its SHAPE, which is a TUPLE
#       - That TUPLES of N non-negative integers that specify the size of each dimension
print("Example Of A 2 Dimensional Array of size 2x3 composed of 4 byte interger elements: \n")
array1 = numpy.array([[1, 2, 3], [4, 5, 6]], numpy.int32)  # 2 Dimensions (Arrays) and 3 Items in each dimension (array)
# The type of items in the array is specified by a separate data-type object (dtype), one of which is associated with each ndarray.
print(f'Showing What Type Of Array This is: {type(array1)}')
print(f'Specific Data Type Object: {array1.dtype}')
# Shape = (# of dimensions(arrays), # of items in each dimension(array) )
print(f"Showing the Shape Of The Array: {array1.shape}")
# The array can be indexed using Python container-like syntax:
# Printing the element of x that is in the second row of the third column
print(f'Printing the element in the second dimension, that in position 3: {array1[1, 2]}\n')
print(f'Printing the elements in the first dimension, thats in position 2: {array1[0, 1]}')
# Slicing can produce various views of the array:
sliceExample1 = array1[:, 1]  # Getting the elements in the second position of both arrays
sliceExample2 = array1[:, 2]  # Getting the elements in the last position of both arrays
print(f'Slicing Example #1: {sliceExample1,}, dtype={sliceExample1.dtype}\n'
      f'Slicing Example #2: {sliceExample2,}, dtype={sliceExample2.dtype}\n')
sliceExample1[0] = 9  # This changes the value in the first example
print(
    f'Slicing Example 3, changing the value of the first slice example: {sliceExample1,},  dtype={sliceExample1.dtype}')
# Constructing arrays
# New arrays can be constructed using these following routines:
'''
From Shape or Value:
empty(shape[, dtype, order, like])
Return a new array of given shape and type, without initializing entries.
'''
# Example Of Empty:
empty1 = numpy.empty([2, 2])
print(f'Example 1 of Returning a new array of given shape and type, without initializing entries: {empty1}')
empty2 = numpy.empty([2, 2], )
print(f'Example 2 of Returning a new array of given shape and type, without initializing entries: {empty2}')
'''
empty_like(prototype[, dtype, order, subok, ...])
Return a new array with the same shape and type as a given array.
'''
# Example of Empty_like:
a = ([1, 2, 3], [4, 5, 6])
print(numpy.empty_like(a))

b = numpy.array([[1., 2., 3.], [4., 5., 6.]])
print(numpy.empty_like(b))
'''
eye(N[, M, k, dtype, order, like])
Return a 2-D array with ones on the diagonal and zeros elsewhere.
            N-int
            Number of rows in the output.
            
            M-int, optional
            Number of columns in the output. If None, defaults to N.
            
            k-int, optional
            Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
            
            dtypedata-type, optional
            Data-type of the returned array.
            
            order{‘C’, ‘F’}, optional
            Whether the output should be stored in row-major (C-style) or column-major (Fortran-style) order in memory.
'''
# Example of eye:
print("Eye Examples: ")
eye1 = numpy.eye(12, 13, dtype=int, k=0)
print(eye1)
print()
eye2 = numpy.eye(8, 8, dtype=str, k=1)
print(eye2)
print()
eye2 = numpy.eye(12, 9, dtype=float, k=6)
print(eye2)
print()
print("numpy.Identity Work:")
'''
identity(n[, dtype, like])
Return the identity array.
    n-int
        Number of rows (and columns) in n x n output.
    
    dtypedata-type, optional
        Data-type of the output. Defaults to float.
    
    likearray_like, optional
        Reference object to allow the creation of arrays which are not NumPy arrays. 
        If an array-like passed in as like supports the __array_function__ protocol, the result will be defined by it. 
        In this case, it ensures the creation of an array object compatible with that passed in via this argument
    
Returns:
    out: ndarray
    n x n array with its main diagonal set to one, and all other elements 0.
'''
print(numpy.identity(3))  # 3 x 3 array with its main diagonal set to one, and all other elements 0.
print(numpy.identity(6))  # 6 x 6 array with its main diagonal set to one, and all other elements to 0
print(
    numpy.identity(12, dtype=str))  # a 12X12 array with it main diagonal set to one, and all the other elements to " "
print(numpy.identity(10,
                     dtype=float))  # a 10X10 array with it main diagonal set to one, and all the other elements to " "

print()

print("Numpy.ones Work: ")
'''
ones(shape[, dtype, order, like])
Return a new array of given shape and type, filled with ones.
Parameters:
        shapeint or sequence of ints
            Shape of the new array, e.g., (2, 3) or 2.
        
        dtypedata-type, optional
            The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.
        
        order{‘C’, ‘F’}, optional, default: C
            Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
        
        likearray_like, optional
            Reference object to allow the creation of arrays which are not NumPy arrays. 
            If an array-like passed in as like supports the __array_function__ protocol, the result will be defined by it. 
            In this case, it ensures the creation of an array object compatible with that passed in via this argument.
Returns: out: ndarray
Array of ones with the given shape, dtype, and order: 
'''
# Examples:
ex1 = numpy.ones(5)  #  Array of ones with the given shape
print(ex1)
print()
ex2 = numpy.ones(5, dtype=int)  #  Array of ones with the given shape, and dtype which is set to integer
print(ex2)
print()
ex3 = numpy.ones((2, 1), dtype=float)  #  Array of ones with the given shape of 2 dimension and 1 element per dimension
print(ex3)
print()
shape = (4, 6)  # Shape = (# of dimensions(arrays), # of items in each dimension(array) )
ex4 = numpy.ones(shape)
print(ex4)
print()
print("Numpy.Ones_Like work: ")
'''
ones_like(a[, dtype, order, subok, shape])
Return an array of ones with the same shape and type as a given array.
Parameters:
    a: array_like
        The shape and data-type of a define these same attributes of the returned array.
    
    dtype: data-type, optional
        Overrides the data type of the result.
    order{‘C’, ‘F’, ‘A’, or ‘K’}, optional
        Overrides the memory layout of the result. 
        ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. 
        ‘K’ means match the layout of a as closely as possible.
    subokbool, optional.
        If True, then the newly created array will use the sub-class type of a, otherwise it will be a base-class array. 
        Defaults to True.

    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order=’K’ and the number of dimensions is unchanged, will try to keep order, otherwise, order=’C’ is implied.
Returns:
out: ndarray
    Array of ones with the same shape and type as a.
'''
ones_likeex1 = numpy.arange(6)  # Return evenly spaced values within a given interval.
print(ones_likeex1)
ones_likeex1 = ones_likeex1.reshape((2, 3))  # Breaks Down The Given Array into a nxm array
print("After the .reshape Function:")
print(ones_likeex1)
print(f'Turning To Ones:\n {numpy.ones_like(ones_likeex1)}')
print("Another reshape call for good measure: ")
ones_likeex2 = numpy.arange(12)
ones_likeex2 = ones_likeex2.reshape((3, 4))
print(ones_likeex2)
print(f'Turning To Ones:\n {numpy.ones_like(ones_likeex2)}')
print()
print("Numpy.zeros Work: ")
'''
zeros(shape[, dtype, order, like])
Return a new array of given shape and type, filled with zeros.
Parameters:
    shapeint or tuple of ints
        Shape of the new array, e.g., (2, 3) or 2.
    
    dtypedata-type, optional
        The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.
    
    order{‘C’, ‘F’}, optional, default: ‘C’
        Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
    
    likearray_like, optional
        Reference object to allow the creation of arrays which are not NumPy arrays. 
        If an array-like passed in as like supports the __array_function__ protocol, the result will be defined by it. 
        In this case, it ensures the creation of an array object compatible with that passed in via this argument.
Returns:
out: ndarray
    Array of zeros with the given shape, dtype, and order.
'''
zeroexample = numpy.zeros(6)
print(f"Zeros Example 1: {zeroexample}")
x = int(input("How many dimension: "))
y = int(input("How many items in each dimension: "))

zeroexample2 = numpy.zeros(shape, dtype=int)
zeroexample3 = numpy.zeros(shape=(2, 4), dtype=str)
print(f"Zeros Example 2:\n {zeroexample2}")
print(f"Zeros Example 3:\n {zeroexample3}")

print("Numpy.Zeros_Like Work:")
'''
zeros_like(a[, dtype, order, subok, shape])
Return an array of zeros with the same shape and type as a given array.
Parameters:
    aarray_like
        The shape and data-type of a define these same attributes of the returned array.
    
    dtypedata-type, optional
        Overrides the data type of the result.
    
    order{‘C’, ‘F’, ‘A’, or ‘K’}, optional
        Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout of a as closely as possible.
    
    subokbool, optional.
        If True, then the newly created array will use the sub-class type of a, otherwise it will be a base-class array. Defaults to True.
    
    shapeint or sequence of ints, optional.
        Overrides the shape of the result. If order=’K’ and the number of dimensions is unchanged, will try to keep order, otherwise, order=’C’ is implied.
Returns:
    out: ndarray
        Array of zeros with the same shape and type as a.
'''
# Example
zeros_like_example1 = numpy.arange(8)
print(f'The basic array: {zeros_like_example1}\n')
zeros_like_example1 = zeros_like_example1.reshape((2, 4))
print(f'Reshaping the basic array into the shape we gave:\n {zeros_like_example1}')
print(f"Turning Everything To Zeros:\n {numpy.zeros_like(zeros_like_example1)}")
print()
zeros_like_example2 = numpy.arange(10)
zeros_like_example2 = zeros_like_example2.reshape((2, 5))
print(zeros_like_example2)
print(numpy.zeros_like(zeros_like_example2))

print()
print("Numpy.full work: ")
'''
full(shape, fill_value[, dtype, order, like])
Return a new array of given shape and type, filled with fill_value.
Parameters:
    shapeint or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    
    fill_valuescalar or array_like
        Fill value.
    
    dtypedata-type, optional
        The desired data-type for the array The default, None, means
        np.array(fill_value).dtype.
    
    order{‘C’, ‘F’}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous (row- or column-wise) order in memory.
    
    likearray_like, optional
        Reference object to allow the creation of arrays which are not NumPy arrays. If an array-like passed in as like supports the __array_function__ protocol, the result will be defined by it. In this case, it ensures the creation of an array object compatible with that passed in via this argument.
'''
full_example1 = numpy.full((2, 2), numpy.inf)
print(full_example1)
full_example2 = numpy.full(shape=(5, 7), fill_value="machew")
print(full_example2)
shape = (3, 8)
full_example3 = numpy.full(shape, fill_value=10, dtype=float)
print(full_example3)
shape23 = (5, 12)
full_example4 = numpy.full(shape23, numpy.NAN, )
print(full_example4)
print()
print("Numpy.full_like work: ")
'''
full_like(a, fill_value[, dtype, order, ...])
Return a full array with the same shape and type as a given array.
Parameters:
    a: array_like
        The shape and data-type of a define these same attributes of the returned array.
    
    fill_value: array_like
        Fill value.
    
    dtype: data-type, optional
        Overrides the data type of the result.
    
    order: {‘C’, ‘F’, ‘A’, or ‘K’}, optional
        Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout of a as closely as possible.
    
    subok: bool, optional.
        If True, then the newly created array will use the sub-class type of a, otherwise it will be a base-class array. Defaults to True.
    
    shape: int or sequence of ints, optional.
        Overrides the shape of the result. If order=’K’ and the number of dimensions is unchanged, will try to keep order, otherwise, order=’C’ is implied.
Returns: out: ndarray
    Array of fill_value with the same shape and type as a
'''
# Examples:
full_like_example1 = numpy.arange(16, dtype=int)
print(full_like_example1)
print(f'Turning To 34s: {numpy.full_like(full_like_example1, 34)}')
full_like_example2 = numpy.arange(12, dtype=float)
print(full_like_example2)
print(f"Filling all the values with '22': {numpy.full_like(full_like_example2, 22)}")
print("----------------------------------------------------------------------------------")
print("Numpy.array Work:")
print()

# One way we can initialize NumPy arrays is from Python lists, using nested lists for two- or higher-dimensional data.
exampleArray = numpy.array([1, 2, 3, 4, 9, 10])
example2Array = numpy.array(([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 135]))
# We can access the elements in the array using square brackets.
# When you’re accessing elements, remember that indexing in NumPy starts at 0.
# That means that if you want to access the first element in your array, you’ll be accessing element “0”.
print(f'Example Array 1: {exampleArray}')
print(f'Printing the first element in array example 1: {exampleArray[0]}')
print(example2Array)
print(f'Printing the first element in the example array 2: {example2Array[0]}')
array1 = numpy.array([1, 2, 3.0])
print(f'Upcasting Example: {array1}')
array2 = numpy.array([[1, 2], [3, 4]])
print(f'An Array With More Than One Dimension:\n {array2}')
array3 = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ndmin=2)
print(f'Minimum Dimensions 2: {array3}')
array4 = numpy.array([1, 2, 3, 4, 5], dtype=complex)
print(f'Array with complex data type: {array4}')
array5 = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=[('a', '<i4'), ('b', '<i4')])
print(f'Array with data-type consisting of more than one elements: {array5}')
print(f'Elements A: {array5["a"]}')

array6 = numpy.array([2, 4, 6, 8, 10])  # This is the exxisting array:
print(f'Creating An array from SUBCLASSES:\n {np.array(array6, subok=True)}')

print("\n\n\nNUMPY.ASARRAY WORK: ")
'''
Convert the input to an array.
Parameters:
a: array_like
    Input data, in any form that can be converted to an array. This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.

dtype: data-type, optional
    By default, the data-type is inferred from the input data.

order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
    Memory layout. ‘A’ and ‘K’ depend on the order of input array a. ‘C’ row-major (C-style), ‘F’ column-major (Fortran-style) memory representation. ‘A’ (any) means ‘F’ if a is Fortran contiguous, ‘C’ otherwise ‘K’ (keep) preserve input order Defaults to ‘K’.

like: array_like, optional
    Reference object to allow the creation of arrays which are not NumPy arrays. If an array-like passed in as like supports the __array_function__ protocol, the result will be defined by it. In this case, it ensures the creation of an array object compatible with that passed in via this argument.
Returns: out: ndarray
        Array interpretation of a. No copy is performed if the input is already an ndarray with matching dtype and order. If a is a subclass of ndarray, a base class ndarray is returned.
'''
print("Converting a list into a array: ")
examplelist = [1,2,3,4,5]
print(numpy.asarray(examplelist))
print("Existing Arrays are not copied: ")
exampleArray2 = numpy.array([1,2])
print(numpy.asarray(exampleArray2) is examplelist)
print(numpy.asarray(examplelist) is examplelist)
print()
print("If dtype is set, array is copied only if dtype does not match:")
a = numpy.array([1, 2], dtype=numpy.float32)
print(numpy.asarray(a, dtype=numpy.float32) is a)
print(numpy.asarray(a, dtype=numpy.float64) is a)
print(numpy.asarray(a, dtype=numpy.int32) is a)
print(numpy.asarray(a, dtype=numpy.int64) is a)
print("Contrary To anarray, ndarray subclasses are not passed through: ")
print(issubclass(numpy.recarray, numpy.ndarray))
print("----------------------------------------------------------------------------------")

breakpoint()

print("----------------------------------------------------------------------------------")
# More information about arrays:
# The NumPy ndarray class is used to represent both matrices and vectors:
# A vector is an array with a single dimension (there’s no difference between row and column vectors),
# A matrix refers to an array with two dimensions

# What are the attributes of an array?
'''
An array is usually a fixed-size container of items of the same type and size. 
The number of dimensions and items in an array is defined by its shape. 
The shape of an array is a tuple of non-negative integers that specify the sizes of each dimension.
'''
print()
# In numpy the dimensions are called axes, meaning if you have a 2d array that looks like this:
TwoDArray = []
evens = []
odds = []
cnt1 = int(0)
cnt2 = int(0)
cnt = int(0)
for nums in range(78):
    nums = random.randint(1, 56)
    cnt += 1
    if (nums % 2 == 0):
        evens.append(nums)
        cnt1 += 1
    elif (nums % 2 != 0):
        odds.append(nums)
        cnt2 += 1
TwoDArray.append(evens)
TwoDArray.append(odds)
print(TwoDArray)
print()
print(f"Your Array Has 2 axes:\n\tThe first axis as a length of {len(evens)}\n\t"
      f"The second axis has a length of {len(odds)}  ")
# How To Create A Basic Array: ===================== CONTINUE HERE TMR
# To create a numpy array you can use the function np.array() #All you need to do to create a simple array is pass a list to it.
# If you choose to, you can also specify the type of data in your list.
lucki = []
for nums in range(10):
    nums = random.randint(1, 456)
    lucki.append(nums)

example3Array = numpy.array(lucki)
print(example3Array)
# Besides creating an array from a sequence of elements, you can easily create an array filled with 0’s:
print("Printing an Array Of 15 0's: ")
np.zeros(15)
print(np.zeros(15))
print()
print(f"Printing an array of ones's: {np.ones(20)}")
'''
Or even an empty array! 
The function empty creates an array whose initial content is random and depends on the state of the memory.
 The reason to use empty over zeros (or something similar) is speed - just make sure to fill every element afterwards!
'''
print(f'Empty Array Of 2 Elements: {np.empty(2)}')

# For 3-D or higher dimensional arrays, the term tensor is also commonly used.


# Create a 2-D array, set every second element in
# some rows and find max per row:

x = np.arange(15, dtype=np.int64).reshape(3, 5)
x[1:, ::2] = -99
x
# array([[  0,   1,   2,   3,   4],
#        [-99,   6, -99,   8, -99],
#        [-99,  11, -99,  13, -99]])

x.max(axis=1)
# array([ 4,  8, 13])

# Generate normally distributed random numbers:
rng = np.random.default_rng()
samples = rng.normal(size=2500)
print(samples)
# ========================================================================================================================
# Example of Using Pandas
'''
The community agreed alias for pandas is pd, 
so loading pandas as pd is assumed standard 
practice for all of the pandas documentation.
'''

'''
To manually store data in a table, create a DataFrame. 
When using a Python dictionary of lists, the dictionary keys will be used as 
column headers and the values in each list as columns of the DataFrame.
'''

'''
The table has 3 columns, each of them with a column label. The column labels are respectively Name, Age and Sex.

The column Name consists of textual data with each value a string, the column Age are numbers and the column Sex is textual data.

'''
df = pandas.DataFrame({"Name": ["Braund, Mr. Owen Harris", "Allen, Mr. William Henry", "Bonnell, Miss. Elizabeth", ],
                       "Age": [22, 35, 58],
                       "Sex": ["male", "male", "female"], })
# Each column in a DataFrame is a Series
'''
A DataFrame is a 2-dimensional data structure that can store 
data of different types (including characters, integers, 
floating point values, categorical data and more) in columns. 
It is similar to a spreadsheet, a SQL table or the data.frame in R
'''
print(df)
print()
print("Print The Ages:")
#printing just the age:
print(df["Age"])
print()
print("Printing The Names: ")
print(df["Name"])
print()
print("Printing The Genders: ")
print(df["Sex"])
print()
# You can create a Series from scratch as well:
# Pandas Series is a one dimensional labeled array capable of holding data of holding data of any type (integer, string, float, python, python objects,etc)

# ALL ARRAYS MUST BE THE SAME LENGTH :

# Pandas Series Examples:
list = [1,2,3,4,5]
exampleseries = pandas.Series(list)
print("Example Series #1: ")
print(list)
print()
# Creating a Series From Scartch:
machew_artist = pandas.Series(["LUCKI", "DESTROY LONEY", "HITLER"], name="Matthews Kings", index=[1, 2, 3])
print(machew_artist)

# If you are familiar with Python dictionaries,
# the selection of a single column is
# very similar to the selection of dictionary values based on the key.

'''
Do something with a DataFrame or Series
I want to know the maximum Age of the passengers

We can do this on the DataFrame by selecting the Age column and applying max():
'''
like = []
for nums in range(12):
    nums = random.randint(1, 466764)
    like.append(nums)
example = pandas.Series(like, name="Random Numbers For Example")
print(example)
print(f"The largest number: {example.max()}")
# Check more options on describe in the user guide section about aggregations with describe
print(example.describe())
print()
print("How do I read and write tabular data?")
print()
'''
pandas provides the read_csv() function to read data stored as a csv file into a pandas DataFrame. 
pandas supports many different file formats or data sources out of the box (csv, excel, sql, json, parquet, …), 
each of them with the prefix read_*.

Make sure to always have a check on the data after reading in the data. 
When displaying a DataFrame, the first and last 5 rows will be shown by default:
'''
titanic = pandas.read_csv("../titanic.csv")
print("Example 1 of Tabular Data")
print(titanic)
policeshootings = pandas.read_csv(
    "../Project 2 - Data Science - Fatal Police Shootings/fatal-police-shootings-data.csv")
print("Example 2 of Tabular Data")
print(policeshootings)
# I want to see the first 8 rows of a pandas DataFrame:
print("--------------------------------------------------------------------------------------------")
print("Printing the first 8 first rows of the titanic data: ")
print(titanic.head(9))
# I want to see the first 20 rows of the fatal police shootings:
print("--------------------------------------------------------------------------------------------")
print("Printing the first 20 rows of the fatal police shooiting data: ")
print(policeshootings.head(21))
print("--------------------------------------------------------------------------------------------")
# To see the first N rows of a DataFrame, use the head()
# method with the required number of rows (in this case 8) as argument.
print()
'''
A check on how pandas interpreted each of the column data types can 
be done by requesting the pandas dtypes attribute:
'''
print("Printing That Data Types Of each Table: ")
print("Titanic.csv")
print(titanic.dtypes)
print()
print("fatal-police-shootings-data.csv")
print(policeshootings.dtypes)
print("--------------------------------------------------------------------------------------------")
print()
#print(titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False))
