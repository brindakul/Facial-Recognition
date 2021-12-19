# Facial-Recognition
Facial Recognition using the models CNN and VGGResnet50 with SVM,Softmax and KNN.
## To install requirements in this project
pip install -r .\scripts\requirements.txt

## To Extract faces from videos
python video2img.py

## To Train CNN-3l
python train.py

## To Fine tune train Resnet-50 of VGG-Face with softmax
python train_VGG.py

## TO Test on test images CNN-3l or VGG-softmax
python predict.py "model"
python predict.py "CNN-3l"
python predict.py "CNN_VGG"

## TO Test on test video CNN-3l or VGG-softmax
python video_pred.py "model"
python video_pred.py "CNN-3l"
python video_pred.py "CNN_VGG"

## To trian SVM or KNN we need to extract features from train images
python feature_extractor.py

## To train and predict SVM
python svm.py 'poly' '500'
python svm.py 'rbg' '100'

## To train and predict KNN
python knn.py k
python knn.py 1
 
