import cv2
import os
from mtcnn.mtcnn import MTCNN
import numpy as np


detector = MTCNN()

path = 'dataset/face_images'
if not os.path.exists(f'{path}/train'):
	os.mkdir(f'{path}/train')
if not os.path.exists(f'{path}/test'):
	os.mkdir(f'{path}/test')

# fname = 'dataset/videos/sreerekha.mp4'
for fname in os.listdir("videos"):
	name = fname.split('.')[0]

	if name not in ['pavithra','lolo','koz','Rawan']:
		continue
	print(name)
	
	if not os.path.exists(f'{path}/train/{name}'):
		os.mkdir(f'{path}/train/{name}')
	if not os.path.exists(f'{path}/test{name}'):
		os.mkdir(f'{path}/test/{name}')	
    

	stream = cv2.VideoCapture(f"videos/{fname}")
	cnt = 0
	while True:
		# grab the frame from the threaded video file stream
		grabbed, frame = stream.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		# resize the frame and convert it to grayscale (while still
		# retaining 3 channels)
		# display a piece of text to the frame (so we can benchmark
		# fairly against the fast method)
		filename = f'{path}/train/{name}/{name}_{cnt}.jpg'
		if cnt%10==0:
			filename = f'{path}/test/{name}/{name}_{cnt}.jpg'

		faces = detector.detect_faces(frame)
		for result in faces:
			x, y, w, h = result['box']
			print(name,x,y,w,h)

			face = frame[y:y + h, x:x + w]
			ht, wd = face.shape[:2]
			if ht>50 and wd>50:
				cv2.imwrite(filename, face)
		# cv2.imwrite(f'img_{cnt}.jpg',face)
		cnt+=1
		cv2.putText(frame,'Testing', (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
		# show the frame and update the FPS counter
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		# fps.update()