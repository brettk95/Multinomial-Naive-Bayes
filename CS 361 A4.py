
# coding: utf-8

# In[1]:


import pandas as pd

trg = pd.read_csv("C:\\Users\\Brett\\Downloads\\abstractclassification\\trg.csv", index_col = 0)

# split each abstract value into a list:
for i in range(len(trg)):
    trg.iloc[i,1] = trg.iloc[i,1].split()


# In[2]:


# split the trg dataframe into 4 dataframes for each class
df = [v for k, v in trg.groupby('class')]

A_df = df[0]
B_df = df[1]
E_df = df[2]
V_df = df[3]

# remove useless words:
list_of_df = [A_df,B_df,E_df,V_df]

remove_more = ['with', 'that', 'from', 'were', 'which', 'that', 'have', 'these', 'been', 'other', 'the', 'this', 'found', 'more', 'three', 'also', 'only', 'open', 'there']

for X_df in list_of_df:
    for i in range(len(X_df)):
        for each in X_df.iloc[i,1]:
            if len(each) < 4 or each in remove_more:
                X_df.iloc[i,1].pop(X_df.iloc[i,1].index(each))


# In[3]:


# find the most common words in each dataframe

words_A = {}
words_B = {}
words_E = {}
words_V = {}

for i in range(len(A_df)):
    for each in A_df.iloc[i,1]:
        if each not in words_A.keys():
            words_A[each] = 1
        else:
            words_A[each] += 1
            
for i in range(len(B_df)):
    for each in B_df.iloc[i,1]:
        if each not in words_B.keys():
            words_B[each] = 1
        else:
            words_B[each] += 1

for i in range(len(E_df)):
    for each in E_df.iloc[i,1]:
        if each not in words_E.keys():
            words_E[each] = 1
        else:
            words_E[each] += 1

for i in range(len(V_df)):
    for each in V_df.iloc[i,1]:
        if each not in words_V.keys():
            words_V[each] = 1
        else:
            words_V[each] += 1


# In[4]:


words_A_sorted = sorted(words_A.items(), key=lambda x: x[1], reverse= True)
words_B_sorted = sorted(words_B.items(), key=lambda x: x[1], reverse= True)
words_E_sorted = sorted(words_E.items(), key=lambda x: x[1], reverse= True)
words_V_sorted = sorted(words_V.items(), key=lambda x: x[1], reverse= True)

A_1000 = []
B_1000 = []
E_1000 = []
V_1000 = []

for i in range(750):
    A_1000.append(words_A_sorted[0:750][i][0])
for i in range(750):
    B_1000.append(words_B_sorted[0:750][i][0])
for i in range(750):
    E_1000.append(words_E_sorted[0:750][i][0])
for i in range(750):
    V_1000.append(words_V_sorted[0:750][i][0])


# In[5]:


# use sets to remove common words that are in all four dictionaries and therefore get a list of features that are composed of only unique words from each class type:

a,b,e,v = set(A_1000),set(B_1000),set(E_1000),set(V_1000)
features = list((a|b|e|v) - ((a&b)|(a&e)|(a&v)|(b&e)|(b&v)|(e&v)) - ((a&b&e)|(a&b&v)|(b&e&v)) - (a&b&e&v))
if "class" in features:
    features.remove("class")
len(features)


# In[6]:


# create a new dataframe with all of the features
df = trg.copy()
for x in features:
    df[x] = 0
    
df = df.drop('abstract',1)
df.head()


# In[7]:


# get the word counts for each class and populate the dataframe

for i in range(4000):
    for word in trg.iloc[i,1]:
        if word in features:
            colnumber = features.index(word) + 1
            df.iloc[i,colnumber] += 1
            
df.head()


# In[54]:


# test out how decent features selection was using the sklearn MultinomialNB package

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

X = df.drop('class',1)
Y = df['class']

clf.fit(X,Y)
predictontrain = clf.predict(X)

error = 0 
for each in list(zip(predictontrain,Y)):
    if each[0] != each[1]:
        error +=1

print(error)
print(100 - error/len(Y)*100)


# In[9]:


# class prior probabilities for each class; P(Ck)

class_priors = {'A':0, 'B':0, 'E':0, 'V':0}
total_class_frequencies = df['class'].value_counts(sort = False)
for key in class_priors:
    class_priors[key] = (total_class_frequencies[key]+1)/(sum(total_class_frequencies)+4)
print(class_priors)


# In[23]:


'''
This might need to be re-done:
P(xi|Ck) = count(xi|Ck)+1 / sum(all x | Ck) + |unique words| .... where |unique words| = len(features)
'''

# create a dictionary for each class to get the prior probabilities for each word given that class - P(xi|Ck)

sorted_df = [v for k, v in df.groupby('class')]
sorted_A = sorted_df[0]
sorted_B = sorted_df[1]
sorted_E = sorted_df[2]
sorted_V = sorted_df[3]

# P(xi|A):
xgivenA = {}
total_word_count_xgivenA = len(features)
for x in features:
    xgivenA[x] = sum(sorted_A[x])+1
    total_word_count_xgivenA += xgivenA[x]

for x in xgivenA:
    xgivenA[x] = xgivenA[x] / (total_word_count_xgivenA+1)
    
# P(xi|B):
xgivenB = {}
total_word_count_xgivenB = len(features)
for x in features:
    xgivenB[x] = sum(sorted_B[x])+1
    total_word_count_xgivenB += xgivenB[x]

for x in xgivenA:
    xgivenB[x] = xgivenB[x] / (total_word_count_xgivenB+1)
    
# P(xi|E):
xgivenE = {}
total_word_count_xgivenE = len(features)
for x in features:
    xgivenE[x] = sum(sorted_E[x])+1
    total_word_count_xgivenE += xgivenE[x]

for x in xgivenE:
    xgivenE[x] = xgivenE[x] / (total_word_count_xgivenE+1)
    
# P(xi|V):
xgivenV = {}
total_word_count_xgivenV = len(features)
for x in features:
    xgivenV[x] = sum(sorted_V[x])+1
    total_word_count_xgivenV += xgivenV[x]

for x in xgivenV:
    xgivenV[x] = xgivenV[x] / (total_word_count_xgivenV+1)


# In[88]:


training = trg.copy()
import math

# caluclate the probability that example is some class Ck given that we have the set of counts:
# use logs
# try code for one training example:

def get_priori(word):
    priori_A = xgivenA[word]
    priori_B = xgivenB[word]
    priori_E = xgivenE[word]
    priori_V = xgivenV[word]
    return priori_A, priori_B, priori_E, priori_V

def get_multiplicative_total_probability_of_x_given_class(x):
    # x = {x1, ... , xn}
    total_A = class_priors['A']
    total_B = class_priors['B']
    total_E = class_priors['E']
    total_V = class_priors['V']
    for xi in x:
        if xi in features:
            total_A *= xgivenA[xi]
            total_B *= xgivenB[xi]
            total_E *= xgivenE[xi]
            total_V *= xgivenV[xi]
    return total_A,total_B,total_E,total_V
     
predicted = []
for i in range(len(training)):
    classes_list = ['A','B','E','V']
    for x in training.iloc[i,1:]:
        pa,pb,pe,pv = get_multiplicative_total_probability_of_x_given_class(x)
        predicted.append(classes_list[([pa,pb,pe,pv].index(max(pa,pb,pe,pv)))])


# In[89]:


error = 0 
for each in list(zip(predicted,Y)):
    if each[0] != each[1]:
        error +=1

print(error)
print(100 - error/len(Y)*100)


# In[82]:


tst = pd.read_csv("C:\\Users\\Brett\\Downloads\\abstractclassification\\tst.csv", index_col = 0)

# split each abstract value into a list:
for i in range(len(tst)):
    tst.iloc[i,0] = tst.iloc[i,0].split()
    
tst.head()


# In[92]:


import math

def get_priori(word):
    priori_A = xgivenA[word]
    priori_B = xgivenB[word]
    priori_E = xgivenE[word]
    priori_V = xgivenV[word]
    return priori_A, priori_B, priori_E, priori_V

def get_multiplicative_total_probability_of_x_given_class(x):
    # x = {x1, ... , xn}
    total_A = class_priors['A']
    total_B = class_priors['B']
    total_E = class_priors['E']
    total_V = class_priors['V']
    for xi in x:
        if xi in features:
            total_A *= xgivenA[xi]
            total_B *= xgivenB[xi]
            total_E *= xgivenE[xi]
            total_V *= xgivenV[xi]
    return total_A,total_B,total_E,total_V
     
predicted = []
for i in range(len(tst)):
    classes_list = ['A','B','E','V']
    for x in tst.iloc[i,0:]:
        pa,pb,pe,pv = get_multiplicative_total_probability_of_x_given_class(x)
        predicted.append(classes_list[([pa,pb,pe,pv].index(max(pa,pb,pe,pv)))])


# In[95]:


import csv
with open('A4.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(pd.Series(predicted))

csvFile.close()
print(pd.Series(predicted))


# In[97]:


a4 = pd.read_csv("C:\\Users\\Brett\\Desktop\\A4.csv", index_col = 0)

