from sklearn.svm import SVC # "Support vector classifier"
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix


val = int(sys.argv[-1])
kernel = sys.argv[-2]

with open('models/features.pickle', 'rb') as f:
    data = pickle.load(f)
face_embbs = data['embeddings']
labels = data["names"]


# kernel = 'poly'
# degree = 100
# gamma = 50
# le = LabelEncoder()
# labels = le.fit_transform(data["names"])
# num_classes = len(np.unique(labels))
# labels = labels.reshape(-1, 1)
# one_hot_encoder = OneHotEncoder(categories='auto')
# labels = one_hot_encoder.fit_transform(labels).toarray()
# f = open('label_encod.pickle', "wb")
# f.write(pickle.dumps(le))
# f.close()

Xtrain, Xtest, ytrain, ytest = train_test_split(face_embbs, labels,test_size=0.2,random_state=42)
# svc = SVC(kernel='rbf', class_weight='balanced')

model_path = 'models/SVM'
if kernel is 'poly':
    svclassifier = SVC(kernel='poly', degree=val)
    svclassifier.fit(Xtrain, ytrain)
    fname = f'svm_model_{kernel}_{val}.pickle'


elif kernel is 'rbf':
    svclassifier = SVC(kernel='rbf', gamma=val)
    svclassifier.fit(Xtrain, ytrain)
    fname = f'svm_model_{kernel}_{val}.pickle'


f = os.path.join(model_path,fname)
pickle.dump(svclassifier, open(f, 'wb'))
 



y_pred = svclassifier.predict(Xtest)
print(confusion_matrix(ytest,y_pred))
print(classification_report(ytest,y_pred))