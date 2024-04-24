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
# Create a proram that stores only the names that are longer than seven characters and have atleast 2 vowel in their name:


def arraywork():
    machewnumbers = numpy.array([[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]], numpy.int64)
    print(f'Print my array using numpy:\n {machewnumbers}\n')
    machewnumbers2 = pandas.DataFrame({"Evens": [2, 4, 6, 8], "Odds": [1, 3, 5, 6]})
    print(f'Print my second array using pandas:\n {machewnumbers2}')

    print("Creating a Series by passing a list of values, letting pandas create a default RangeIndex.")
    machewnumbers3 = pandas.Series([1,3,5,numpy.nan,6,7])
    print(machewnumbers3)
    print("Creating a DataFrame by passing a NumPy array with a datetime index using date_range() and labeled columns:")
    machewnumbers4_dates = pandas.date_range("20240414",periods=6)
    print(machewnumbers4_dates)
    machewnumbers5_dates = pandas.DataFrame(numpy.random.randn(6, 4), index=machewnumbers4_dates, columns=list("ABCD"))
    print(machewnumbers5_dates)
    print(machewnumbers5_dates.head())
    print()
    print("Creating a DataFrame by passing a dictionary of objects where the keys are the column labels and the values are the column values.")
    machewnumbers6 = pandas.DataFrame({"A": 1.0,
                                       "B": pandas.Timestamp("20140102"),
                                       "C": pandas.Series(1,index=list(range(4)),dtype="float32"),
                                        "D": numpy.array([3]*4,dtype="int32"),
                                       "E": pandas.Categorical(["test","train","test","train"]),
                                       "F": "foo"})
    print(machewnumbers6)
    print()
    print("Creating A Datafrom of 5 artist and their individual top 5 songs: ")
    machewArtist1 = pandas.DataFrame({"LUCKI":["GOODFELLAS","All love","Newer Me","LOVE IS WAR","Lil Ol Me"],
                                      "Baby Smoove":["DX","I Dare you","Sleep Walking Pt.2","Instagram", "Freestyle"],
                                      "Warhol.SS":["SECOND NATURE", "MONEY TALK","Big 32","Count Up (unreleased)","Cetified Raq Baby"],
                                      "Yeat":["Double","Money Twerk","Type Money","Demon Tiez","Kant Die"],
                                      "Destroy Lonely":["Worth It","red dead","dream about me.","ONTHETABLE","BLITZ"]},index=[1,2,3,4,5]) # peep the index, i had to list out 1-5 because there are 5 dimensions in this array
    print(machewArtist1)
    print("The columns of the resulting DataFrame have the same dtypes:")
    print(machewArtist1.dtypes)
    first3rowsfrommachewartist = machewArtist1.head(3)
    print("The First 3 Rows of the Machew Artist Array: ")
    print(first3rowsfrommachewartist)
    last3rowsfrommachewartist = machewArtist1.tail(3)
    print()
    print("The Last 3 Rows of the Machew Artist Array: ")
    print(last3rowsfrommachewartist)
    print()
    print("Printing the indexs: ")
    print(machewArtist1.index)
    print()
    print()

    print()

#arraywork()
'''
df = pandas.DataFrame({"Name": ["Braund, Mr. Owen Harris", "Allen, Mr. William Henry", "Bonnell, Miss. Elizabeth", ],
                       "Age": [22, 35, 58],
                       "Sex": ["male", "male", "female"], })
'''
# Create a proram that stores only the names that are longer than seven characters and have atleast 2 vowel in their name:
print(chr(33))
print("dd")
print(ord('x'))


