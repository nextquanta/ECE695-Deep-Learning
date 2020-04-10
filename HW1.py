# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:22:20 2020

@author: risij
"""

import random
import string
import sys
import pdb
random.seed(0)

class People:
    def __init__(self,first_names,middle_names,last_names,print_format):
        self.first_names = first_names
        self.middle_names = middle_names
        self.last_names = last_names
        self.print_format=print_format
        self.index = -1
      
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index+=1
        if(self.print_format=="first_name_first"):
            if(self.index<len(self.first_names)):
                return " ".join([self.first_names[self.index],self.middle_names[self.index],self.last_names[self.index]])
            else:
                raise StopIteration
        elif(self.print_format=="last_name_first"):
            if(self.index<len(self.middle_names)):
                return " ".join([self.last_names[self.index],self.first_names[self.index],self.middle_names[self.index]])
            else:
                raise StopIteration
        elif(self.print_format=="last_name_with_comma_first"):
            if(self.index<len(self.last_names)):
                return self.last_names[self.index]+", "+self.first_names[self.index]+" "+self.middle_names[self.index]
            else:
                raise StopIteration     
        else:
            sys.stderr.write("Invalid Option Provided")
    
    def __call__(self):
        for items in sorted(self.last_names):
            print(items)
    
    next = __next__        

class PeopleWithMoney(People):
    def __init__(self,first_names,middle_names,last_names,print_format,wealth):
        People.__init__(self,first_names,middle_names,last_names,print_format)
        self.wealth = wealth
    
    def __next__(self):
        try:
            return People.__next__(self)+" "+ str(self.wealth[self.index])
        except:
            raise StopIteration           
    
    def __call__(self):
        sorted_info=sorted(zip(self.first_names,self.middle_names,self.last_names,self.wealth),key = lambda x:x[3])

        for item in sorted_info:
            print(item[0],end=" ")
            print(item[1],end=" ") 
            print(item[2],end=" ") 
            print(item[3]) 
            
    next=__next__

first_names = ["".join([random.choice(string.ascii_lowercase) for i in range(5)]) for j in range(10)]
middle_names = ["".join([random.choice(string.ascii_lowercase) for i in range(5)]) for j in range(10)]
last_names = ["".join([random.choice(string.ascii_lowercase) for i in range(5)]) for j in range(10)]
#wealth = random.sample(range(0,1000),10)
wealth = [random.randint(0,1000) for i in range(10)]


P1 = People(first_names,middle_names,last_names,"first_name_first")
P1_iter = iter(P1)
for i in range(10):
    print(P1_iter.next())        
        
print()

P2 = People(first_names,middle_names,last_names,"last_name_first")
P2_iter = iter(P2)
for i in range(10):
    print(P2_iter.next())   

print()


P3 = People(first_names,middle_names,last_names,"last_name_with_comma_first")
P3_iter = iter(P3)
for i in range(10):
    print(P3_iter.next())   
    
print()
    
P1()

print()

      
PM = PeopleWithMoney(first_names,middle_names,last_names,"first_name_first",wealth)
PM_iter = iter(PM)
for i in range(10):
    print(PM_iter.next())   

print()

PM()



#first_name_array = P1.first_names
#middle_name_array=P1.middle_names
#last_name_array = P1.last_names





