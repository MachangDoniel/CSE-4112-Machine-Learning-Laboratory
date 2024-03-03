# -*- coding: utf-8 -*-
"""1907121 lab 01.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YAYOiWiNR8VB5HZNR4W7KfNDmwpuWNaw

**Introduction to Python**
"""

var = 12234723424893274838423847328838328*pow(1222,121)

print(var)
print(type(var))
print(len(str(var)))

x = input("Enter a value: ")

print(int(x)+10)

"""Conditional Statement"""

a = 500
if (a<200):
  print("a is smaller than 200")
elif (a==200):
  print("a is equal to 200")
else:
  print("x is greater than 200")
  if a<1000 and a>300:
    print("nested")

"""Array"""

arr= [1,5,2,3,4]
print(arr)
for i in range(len(arr)):
  print(i,":", end=" ")
  print(arr[i])
for i in arr:
  print(i, end=" ")
print()
arr.pop(1)
print(arr)
arr.insert(1,5)
print(arr)
arr.append(19)
print(arr)
arr.sort()
print(arr)
arr.sort(reverse=True)
print(arr)

"""List"""

ll = [4,1,2,3.5]
print(ll)
print(ll[0])

"""Comment
single by #
multiple by start with ''' hi ''' or ''' hello '''
"""

# print(ll)
print(ll[0])
''' print(ll[1])
print(ll[2]) '''
print(ll[3])

"""Data *structure*"""

ll.append(20)
print(ll)

ll.sort()
print(ll)
ll.sort(reverse=True)
print(ll)

"""Map or Dictionary"""

dictionary = {"pkd":["alu","potol"], "doniel":121, "nd":{1:2,2:3,3:4}}
print(dictionary)
print(dictionary["pkd"])
print(dictionary["doniel"])
print(dictionary["nd"][2])

"""Loop (while, for)"""

i = 0
while i<10:
  print(i, end = " ") # by default end="newline"
  i+=1

for i in 1,2,3,4,5: # works on data structure like list
  print(i, end = " ")
print()
for i in ll:
  print(i, end = " ")
print()
for i in dictionary:
  print(i, end = " ")
print()
for i in dictionary:
  print(dictionary[i], end = " ")
print()
for i in range(20):
  print(i, end = " ")

"""Function"""

def function (a,b):
  return a+1,b+2
def add (a,b):
  return a+b

a,b = function(10,20)
print(a,b)
print(add(a,b))

"""random number"""

import random
for i in range(10):
  print(random.randint(0,10), end = " ")

"""Numpy array package"""

import numpy
mat1 = numpy.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
mat2 = numpy.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
mat3=mat1+mat2
print(mat3)
mat4=mat1-mat2
print(mat4)
mat5=mat1*mat2
print(mat5)
mat6=mat1.dot(mat2)
print(6)

"""Pandas package
data manipulation, handling, preprocessing
"""

import pandas as pd
data = pd.read_excel("/content/1907121.xlsx")
print(data)

"""Plot with pandas
(ctrl +space) for suggestion
matplotlib for plotting
"""

data.plot(kind='line',x='Serial',y="CGPA")

"""workbook
xlxswriter, xlrd used by sir
openpyxl, xlwings used by me
"""

cgpa_column=data['CGPA']
print(cgpa_column)
# print()
# print(cgpa_column.mean())