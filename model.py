#dataset,feature engineering,ind feature variable,
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle  #it helps to dumb your model with some extension
#from sklearn.model_selection import train_test_split

dataset=pd.read_csv("C:/Users/Divya/Downloads/insurance.csv")
#print(dataset)


#Ind and dep feature
X=dataset.iloc[:,:5]


def convert_to_int(word):
    word_dict={'male':1,'female':0}
    return word_dict[word]

def convertto_int(word1):
    worddict={'yes':1,'no':0}
    return worddict[word1]

X['sex']=X['sex'].apply(lambda x:convert_to_int(x))
X['smoker']=X['smoker'].apply(lambda x:convertto_int(x))



#dep feature
y=dataset.iloc[:,-1]

#divide the data
#x_train,y_train,x_test,y_test=train_test_split(X,y,random_state=5)


#model 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#fitting model with training data
regressor.fit(X,y)
#saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))  #this file will be deployed it on heroku (i.e PAAS)
