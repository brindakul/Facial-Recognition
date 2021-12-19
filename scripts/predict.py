import cv2
import os
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


dim = (224,224)
model_name = sys.argv[-1]
model = load_model(f'models/{model_name}/model.h5')
LABELS = open(f"models/{model_name}/LABELS.txt").read().split('\n')

print(model.summary())
ytest = []
ypred = []
path  = 'dataset/face_images/test'
for gt in os.listdir(path):
  # matrix = {'TP':0, 'FP':0, 'TN': 0, 'FN':0}
  for imgf in os.listdir(os.path.join(path,gt)):
    img = cv2.imread(os.path.join(path,gt,imgf))
    im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    im_rgb = cv2.resize(im_rgb, dim, interpolation = cv2.INTER_AREA)
    im_rgb = np.array(im_rgb)/255
    im = np.expand_dims(im_rgb, axis=0)
    res = model.predict(im)
    pred = LABELS[np.argmax(res[0])]
    ytest.append(gt)
    ypred.append(pred)
    cv2.imwrite()
  #   if pred == gt:
  #     matrix['TP'] +=1
  
  # print(gt,matrix['TP'])
  # print('Accuracy: ',matrix['TP']/len(os.listdir(os.path.join(path,gt))))
ytest = np.array(ytest)
ypred = np.array(ypred)
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))  
# print(gt,l)



