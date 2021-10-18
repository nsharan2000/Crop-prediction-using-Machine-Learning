#importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Reading the csv file
data=pd.read_csv('cpdata.csv')
print(data.head(1))

#Creating dummy variable for target i.e label
label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('The data present in one row of the dataset is')
print(data.head(1))
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values

#Dividing the data into training and test set
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Decision Tree classifier
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()

#Fitting the classifier into training set
clf.fit(X_train,y_train)
pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,pred)
print("The accuracy of this model is: ", a*100)

#Using firebase to import data to be tested
from firebase import firebase
firebase =firebase.FirebaseApplication('https://cropit-eb156.firebaseio.com/')
tp=firebase.get('/Realtime',None)

ah=tp['Air Humidity']
atemp=tp['Air Temp']
shum=tp['Soil Humidity']
pH=tp['Soil pH']
rain=tp['Rainfall']


l=[]
#temperature comes before ph in the dataset. these results will aslo wildly affect the output
l.append(atemp)
l.append(ah)
l.append(pH)
l.append(rain)
predictcrop=[l]

#the coloumns need to be in the same order as the datapoints for correct input
crops = np.array(data.iloc[: ,4:].columns);


inputval = sc.transform([[17.47500984,	16.54314829,  6.18042747,	93.35034262]]) #sample values for chickpeas
#predicting the crop
predictions = clf.predict(inputval)
#you forgot to transform the values before passing it to the predictor

count=0
for i in range(0,30):
    if(predictions[0][i]==1):
        c=crops[i]
        count=count+1
        break;
    i=i+1
if(count==0):
    print('The predicted crop is %s'%cr)
else:
    print('The predicted crop is %s'%c)

#Sending the predicted crop to database
cp=firebase.put('/croppredicted','crop',c)
