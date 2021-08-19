# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:15:55 2021
fake news detection
@author: Shruti Dhave
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk 
from tkinter import ttk  

wc=tk.Tk()
wc.geometry("400x500")
wc.title("Fake news detection window")


L=tk.Label(wc,text="FAKE NEWS DETECTION USING ML",font=('arial',20),bg="pink")
L.grid(row=0,column=3)

L1=tk.Label(wc,text="Enter News",font=('arial',18),bg="pinK",bd=10)
L1.grid(row=1,column=2,padx=20,pady=15)


#tk.Entry(wc,height=5, width = 30)
E1= tk.Text(wc,height=4,width=30)#to give text boxes
E1.grid(row=1,column=3)

#importing dataset
df_fake = pd.read_csv("D:\Ml Data\Fake1.csv")
df_true = pd.read_csv("D:\Ml Data\True1.csv")
df_fake.head()
df_true.head(5)

#importing column as a class feature
df_fake["class"] = 0
df_true["class"] = 1
df_fake.shape, df_true.shape

#Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(79,60,-1):
    df_fake.drop([i], axis = 0, inplace = True)
       
df_true_manual_testing = df_true.tail(10)
for i in range(79,69,-1):
    df_true.drop([i], axis = 0, inplace = True)
    
df_fake.shape, df_true.shape
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1
df_fake_manual_testing.head(10)
df_true_manual_testing.head(10)
df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")
df_merge = pd.concat([df_fake, df_true], axis =0 )
df_merge.head(10)
df_merge.columns
df = df_merge.drop(["title", "subject","date"], axis = 1)
df.isnull().sum()

df = df.sample(frac = 1)
df.head()
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
df.columns
df.head()

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#converting text to vectors
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#logistic regression
LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)
classification_report(y_test, pred_lr)

#Initialize a PassiveAggressiveClassifier
PAC=PassiveAggressiveClassifier(max_iter=50)
PAC.fit(xv_train,y_train)
# Predict on the test set and calculate accuracy
pred_pac=PAC.predict(xv_test)
PAC.score(xv_test, y_test)
classification_report(y_test, pred_pac)

#Gradient Boosting Classifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)
classification_report(y_test, pred_gbc)

#Random Forest Classifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)
classification_report(y_test, pred_rfc)

#Model Testing
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"
     
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_PAC = PAC.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)    
    return print("\n\nLR Prediction: {} \nPAC Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                                               output_lable(pred_PAC[0]), 
                                                                                                             output_lable(pred_GBC[0]),
                                                                                                           output_lable(pred_RFC[0])))
global news
def all():        
        news = E1.get("1.0",'end-1c')
        manual_testing(news)        
        Lbl.configure(foreground = 'red') 
        action.configure(text = "Clicked")  
              
Lbl = ttk.Label(wc,text = "Click Button to Check")  
Lbl.grid(row=8,column=3)# Click event  
    
# Adding Button  
action = ttk.Button(wc, text = "Click to check", command = all )  
action.grid(row=6,column=3)  

wc.mainloop() 
