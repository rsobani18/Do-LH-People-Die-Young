#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Do Lefthanded people really die young? According to a 1991 study, left-handed people die
# on average, 9 years earlier than their counterparts. 
#Using pandas and Bayesian statistics to analyze the correlation between age of death and being left or right handed

#Importing pandas and matplotlib.pyplot
import pandas as pd
import matplotlib.pyplot as plt

#Load data from link into dataframe
link_1 = "https://gist.githubusercontent.com/mbonsma/8da0990b71ba9a09f7de395574e54df1/raw/aec88b30af87fad8d45da7e774223f91dad09e88/lh_data.csv"
lefty = pd.read_csv(link_1, sep = ",")

#make a plot of M & F left-handed people rate vs. age
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots()
ax.plot("Age", "Female", data = lefty, marker = "+" )
ax.plot("Age", "Male", data = lefty, marker = "o")
ax.legend()
ax.set_xlabel("Age")
ax.set_ylabel("Gender")


# In[2]:


#To get a rate of left-handed people over time, find a single average rate for M & F
#Add a new column to lefty for birth year
lefty["Birth_year"] = 1986 - lefty["Age"]

#new column for mean of M&F and columns
lefty["Mean_lefty"] = lefty[["Male", "Female"]].mean(axis = 1)

#create plot using new columns
fig, ax = plt.subplots()
ax.plot("Birth_year", "Mean_lefty", data = lefty)
ax.set_xlabel("Birth Year")
ax.set_ylabel("Percentage left-handed")


# In[5]:


#probability of dying at certain age given that left handed not = to probability of being left-handed given that you died at a certain age
#Need to use Bayes' thereom to find probability of dying at a certain age given that one is left-handed. Also need for RH.
#import numpy package
import numpy as np

def lefty_prob_given_A (ages_of_death, study_year= 1990):
    """ P(Left-handed | ages of death), calculated based on the reported rates of left-handedness.
    Inputs: numpy array of ages of death, study_year
    Returns: probability of left-handedness given that subjects died in `study_year` at ages `ages_of_death`""" #preloaded code in Data Camp project

    #Use the mean of 10 last and 10 first points for lefthandedness rates before and after the start
    beg_1900s = lefty["Mean_lefty"][-10:].mean()
    end_1900s = lefty["Mean_lefty"][:10].mean()
    middle_rates = lefty.loc[lefty["Birth_year"].isin(study_year - ages_of_death)]["Mean_lefty"]
    youngest = study_year - 1986 + 10 #youngest age is 10
    oldest = study_year - 1986 +86 #oldest age is 86

    #create empty array to store results
    empty = np.zeros(ages_of_death.shape)
    #rate of left-handedness for people of ages 'ages of death'
    empty[ages_of_death > oldest] = beg_1900s/100
    empty[ages_of_death < youngest] = end_1900s/100
    empty[np.logical_and((ages_of_death <= oldest), (ages_of_death >= youngest))] = middle_rates/100

    return empty


# In[9]:


#data used as probability distribution that gives probability of dying at a certain age
#data on distribution of death for US in 1999
link2 = "https://gist.githubusercontent.com/mbonsma/2f4076aab6820ca1807f4e29f75f18ec/raw/62f3ec07514c7e31f5979beeca86f19991540796/cdc_vs00199_table310.tsv"

#read file into dataframe
death_data = pd.read_csv(link2, sep = '\t',skiprows=[1])
#get rid of the NA values
death_data= death_data.dropna(subset = ["Both Sexes"])

#plot age vs people who died from both M&F
fig, ax = plt.subplots()
ax.plot("Age", "Both Sexes", data = death_data, marker = "o")
ax.set_xlabel("Age")
ax.set_ylabel("Number of people who died")


# In[10]:


#Finding the overall probability of left-handedness
def Prob_lh(death_data, study_year = 1990): # sum over Prob_lh for each age group
    """ Overall probability of being left-handed if you died in the study year
    Prob_lh = P(LH | Age of death) P(Age of death) + P(LH | not A) P(not A) = sum over ages
    Input: dataframe of death distribution data, study year
    Output: P(LH), a single floating point number """
    prob_list = death_data["Both Sexes"]*lefty_prob_given_A(death_data['Age'], study_year) # multiply number of dead people by P_lh_given_A
    p = np.sum(prob_list) # calculate the sum of prob_list
    return p/np.sum(death_data["Both Sexes"]) # normalize to total number of people (sum of death_distribution_data['Both Sexes'])

print(Prob_lh(death_data))


# In[14]:


#Probability of being age 'A' at death
def death_age_lh(ages_of_death, death_data, study_year = 1990):
    """ The overall probability of being a particular `age_of_death` given that you're left-handed """
    
    P_A = death_data['Both Sexes'][ages_of_death]/np.sum(death_data['Both Sexes'])
    P_left = Prob_lh(death_data, study_year) 
    P_lh_A = lefty_prob_given_A(ages_of_death, study_year)
    
    return P_lh_A*P_A/P_left


# In[12]:


def death_age_rh(ages_of_death, death_data, study_year = 1990):
    """ The overall probability of being a particular `age_of_death` given that you're right-handed """
    
    P_A = death_data['Both Sexes'][ages_of_death]/np.sum(death_data['Both Sexes'])
    P_right = 1- Prob_lh(death_data, study_year) # P_right = 1 - P_left
    P_rh_A = 1 - lefty_prob_given_A(ages_of_death, study_year) 
    
    return P_rh_A*P_A/P_right


# In[15]:


#PLotting probability of above for range of ages of death
ages = np.arange(6, 115, 1) 

# probability of being LH or RH for each 
LH_probability = death_age_lh(ages, death_data)
RH_probability = death_age_rh(ages, death_data)

# plot the two probs vs. age
fig, ax = plt.subplots() 
ax.plot(ages, LH_probability, label = "Left-handed")
ax.plot(ages, RH_probability, label = "Right-handed")
ax.legend()
ax.set_xlabel("Age at death")
ax.set_ylabel(r"Probability of being age A at death")


# In[ ]:




