'''
Matthew Kilpatrick
4/13/24
Geeks For Geeks
'''
import pandas
import numpy
import random
from tabulate import tabulate
# You can create a Series from scratch as well:
# Pandas Series is a one dimensional labeled array capable of holding data of holding data of any type (integer, string, float, python, python objects,etc)

# ALL ARRAYS MUST BE THE SAME LENGTH :

# Pandas Series Examples:
print("Example Series #1: In order to create a series from list, we have to first create a list after that we can create a series from list. ")
list1 = [1, 2, 3, 4, 5]  # Creating a simple LIST
print(f"This is the list: {list1}")
exampleseries = pandas.Series(list1, index=[1,2,3,4,5], dtype=float) # Turing the simple list into a series and peep the float dtype
print("This is the same list above but turned into SERIES: ")
print(exampleseries)
print()
print("Example Series #2: Creating a series from array:\n "
      "\tIn order to create a series from array, we have to import a numpy module and have to use array() function. \n")
data = numpy.array(['M', 'a', 'c', 'h', 'e', 'w'], dtype=str)  # Creating a simple ARRAY
print(f'This is the simple ARRAY: {data}\n')
exampleseries2 = pandas.Series(data)
print("This is the same list above but turned into a SERIES: ")
print(exampleseries2)
print()
print("Accessing element of Series")
'''
Accessing Element from Series with Position : 
In order to access the series element refers to the index number. 
Use the index operator [ ] to access an element in a series. 
The index must be an integer. 
In order to access multiple elements from a series, we use Slice operation.
'''
print("Example Series #3: Accessing the first 3 elements of a SERIES")
print(exampleseries[:3])
print()
print("Accessing Element Using Label (index) :")

# In order to access an element from series, we have to set values by index label. A Series is like a fixed-size dictionary in that you can get and set values by index label.
# In order to access multiple elements from a series, we use Slice operation.
# To print elements from beginning to a range use [:Index], to print elements from end-use [:-Index], to print elements from specific Index till the end use [Index:], to print elements within a range, use [Start Index:End Index] and to print whole Series with the use of slicing operation, use [:].

data2 = ["LUCKI", "Machew", "Raven Mills", "Hitler", "Jewish People", "Drugs"]
example3series = pandas.Series(data2, index=[1,2,3,4,5,6])
print(example3series)
# In order to access the series element refers to the index number. Use the index operator [ ] to access an element in a series. The index must be an integer.
print(f'Printing the data located in postion {5}: {example3series[5]}\n')
# Further, to print the whole Series in reverse order, use [::-1].
print(f"Printing the series in reverse: \n{example3series[::-1]}\n")
#  [Start Index:End Index]
print(f"Printing Elements with range {3} - {5}:\n{example3series[3:5]}\n")
# to print elements from end-use [:-Index]
print(f"Printing the Elements from the End to Third Element:\n{example3series[-4:]}")
# To print elements from beginning to a range use [:Index]
print(f"Printing Elements from beginning to 4:\n{example3series[:4]}")
examplelist = ['Molly', "Xanax", "Ketamine", "MDMA", "Percocet", "Oxycontin", "PCP", "Weed", "LSD", "Ecstasy", "DMT"]
ex4array = numpy.array(examplelist, numpy.str_)  # with numpy.str
ex5array = numpy.array(examplelist,dtype=str)  # with dtype = str
print(ex4array.shape)
ex4sdataframe = pandas.DataFrame(ex4array,index=[1,2,3,4,5,6,7,8,9,10,11])
ex4series = pandas.Series(ex4array,index=[1,2,3,4,5,6,7,8,9,10,11],dtype=str)
print("====================================================================================================================")
print(ex4array)
print()
print("Printing The DataFrame version of the list in the tabulate package: ")
print(tabulate(ex4sdataframe))
print()
names = ["Machew","Raven","Maddie","CJ","John","Chris","Jasmine","Alexis Chantel","LUCKI","IBN"]
namesarray = numpy.array(names)
ages = [21,21,22,21,20,19,23,22,21,19]
agearray = numpy.array(ages)
majors = ["Computer Science","Architecture","Psychology","Cyber Security","SportsManagement","Journalism","Engineering","Business","Graphic Design","Math"]
majorsarray = numpy.array(majors)
dataframeex1 = pandas.DataFrame({"Names": namesarray,
                                 "Ages": agearray,
                                 "Majors": majorsarray},
                                index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
seriesexample3 = pandas.Series({"Names": names,"Ages":ages,"Majors":majors},index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataframeex1.shape)
print(dataframeex1)
print("Accessming Multiples Elements Using the INDEX LABEL: ")
data12 = numpy.array(['Machew', "Raven", "LUCKI", "Yung Bans", "Future"],dtype=str)
series12 = pandas.Series(data12, index=[1,2,3,4,5],dtype=str)
print(f"Printing Elements in postion 2,3, and, 5:\n{series12[[2,3,5]]}")
print(f"Printing Multiple Elements From The Names Ages And Majors Array:\n{dataframeex1[:3]}")
print("===============================================================================================")
print("Accessing A Multiple Element Using Index Label in the NBA.csv file: ")
# Making a dataframe:
print()
dataframeexample1 = pandas.read_csv("nba.csv", header=0)
print("Printing The NBA.CSV FIle IN STANDARD DATAFRAME FORM: ")
print(dataframeexample1)
print()
print(f"NBA.CSV file shape: {dataframeexample1.shape}")
print()
print("Creating A Pandas Series Of The Names Column from the Dataframe: ")
nbaseriesofNames = pandas.Series(dataframeexample1["Name"])
nbaseriesofTeams = pandas.Series(dataframeexample1["Team"])
print(f'Printing The First 12 Names Of The Nba.CSV File:\n{nbaseriesofNames.head(12)}')
print()
print(f"Printing The First 13 Team Names from the teams column Of The NBA.CSV File:\n{nbaseriesofTeams.head(38)}")
print(f"Printing The random names from the file:\n{nbaseriesofNames[[68,40,438]]}")
first100teams = nbaseriesofTeams.head(100)
print(f"Printing Teams Column Within the Range Of 34-75:\n{first100teams[35:75+1]}")

# Think of a SERIES as a column of data, such as a collection of a single variable:
# A DATAFRAME is an object for storing related columns of data
# IN essence a dataframe in pandas is several columns , one for each variable
