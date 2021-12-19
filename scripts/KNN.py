from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
with open('models/features.pickle', 'rb') as f:
    data = pickle.load(f)
face_embbs = data['embeddings']
labels = data["names"]

k = sys.argv[-1]
Xtrain, Xtest, ytrain, ytest = train_test_split(face_embbs, labels,test_size=0.2,random_state=42)
# svc = SVC(kernel='rbf', class_weight='balanced')

knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean',p=3)
knn.fit(Xtrain, ytrain)

model_path = 'models/KNN'
fname = f'KNN_model_{k}.pickle'
f = os.path.join(model_path,fname)
pickle.dump(knn, open(f, 'wb'))
 


y_pred = knn.predict(Xtest)
print(confusion_matrix(ytest,y_pred))
print(classification_report(ytest,y_pred))