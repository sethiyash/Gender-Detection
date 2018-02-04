import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import mglearn 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

mydata = pd.read_csv("voice.csv");

#print (mydata.head(0))
#print (mydata.shape)

#Histogram between target and independent variable

male  = mydata.loc[mydata['label']=='male']
female = mydata.loc[mydata['label'] == 'female']

'''
for i in range(20):
	trace1 = go.Histogram(x = male.ix[:,i])
	trace2 = go.Histogram(x = female.ix[:,i])
	data = [trace1,trace2]
	layout = go.Layout(barmode = 'overlay')
	fig = go.Figure(data = data,layout=layout)
	py.iplot(fig,filename = 'hist')
'''

fig, axes = plt.subplots(10, 2, figsize=(10,20))
ax = axes.ravel()
for i in range(20):
    ax[i].hist(male.ix[:,i], bins=20, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(female.ix[:, i], bins=20, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(list(male)[i])
    ax[i].set_yticks(())
    
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["male", "female"], loc="best")
fig.tight_layout()


#prepare data for modeling
mydata.loc[:,'label'][mydata['label'] == "male"] = 0
mydata.loc[:,'label'][mydata['label'] == "female"] = 1

#print (mydata.head(1))
mydata_train,mydata_test = train_test_split(mydata,random_state = 0,test_size = 0.2)

scalar = StandardScaler()

scalar.fit(mydata_train.ix[:,0:20])

feature = list(mydata)
#print(feature)
X_train = scalar.transform(mydata_train.ix[:,0:20])
X_test = scalar.transform(mydata_test.ix[:,0:20])
y_train = list(mydata_train['label'].values)
y_test = list(mydata_test['label'].values)

clf = DecisionTreeClassifier(random_state = 0).fit(X_train,y_train)
print ("Decision Tree")
print("Accurracy on Training Set: {:.3f}".format(clf.score(X_train,y_train)))
print("Accurracy on Test Set: {:.3f}".format(clf.score(X_test,y_test)))

# print(feature[0:20])
# print(X_test[1])
# print(y_test[1])

new_y_train = []
for x in y_train:
	if(x == 0):
		new_y_train.append("Male");
	else: 
		new_y_train.append("Female");
tree.export_graphviz(clf,out_file='tree.dot', feature_names=feature[0:20],class_names = new_y_train,filled= True,rounded = True, special_characters = True)


