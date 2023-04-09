import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5")
camera = cv2.VideoCapture(0)
while True:
	status , frame = camera.read()
	if status:
		frame = cv2.flip(frame , 1)
		img = cv2.resize(frame ,(224,224))
		test_img = np.array(img,dtype=np.float32)
		test_img = np.expand_dims(test_img,axis = 0)
		normalized_img = test_img/255.0
		prediction = model.predict(normalized_img)
		print("Prediction",prediction)
		rock = int(prediction[0][2]*100)
		paper = int(prediction[0][0]*100)
		scissor = int(prediction[0][1]*100)
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")
		cv2.imshow('feed' , frame)
		code = cv2.waitKey(1)
		if code == 32:
			print("Closing")
			break
camera.release()
cv2.destroyAllWindows()