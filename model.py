import pandas as pd
import numpy as np

df = pd.read_csv('BankNote_Authentication - 2024-08-12T114241.106.csv')
df.head()

x =df.drop('class', axis=1)
y = df['class']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)


import pickle
with open('model.pkl','wb') as model_file:     #wb means write and buffering model
    pickle.dump(clf,model_file)
    
