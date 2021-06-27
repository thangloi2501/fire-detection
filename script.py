import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the saved model
model = tf.keras.models.load_model('./InceptionV3.h5')
# video = cv2.VideoCapture(0)
#
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# while True:
#     _, frame = video.read()
#     # Convert the captured frame into RGB
#     im = Image.fromarray(frame, 'RGB')
#     # Resizing into 224x224 because we trained the model with this image size.
#     im = im.resize((224, 224))
#     img_array = image.img_to_array(im)
#     img_array = np.expand_dims(img_array, axis=0) / 255
#     probabilities = model.predict(img_array)[0]
#     # Calling the predict method on model to predict 'fire' on the image
#     prediction = np.argmax(probabilities)
#     print(probabilities, " >>>>>>>>>>>>>>>> ", prediction)
#     # if prediction is 0, which means there is fire in the frame.
#     if prediction == 0:
#         # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         cv2.putText(frame, 'FIRE!!!', (10, 450), font, 2, (0, 0, 255), 5, cv2.LINE_AA)
#     cv2.imshow("Capturing", frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()

im = image.load_img('./test/house2.jpg')
# Resizing into 224x224 because we trained the model with this image size.
im = im.resize((224, 224))

img_array = image.img_to_array(im)
img_array = np.expand_dims(img_array, axis=0) / 255

plt.imshow(im)
plt.show()
# input("Press Enter to continue...")

probabilities = model.predict(img_array)[0]
print("probabilities = ", probabilities)
prediction = np.argmax(probabilities)
print("prediction = ", prediction)
if prediction == 0:
    print("There is a FIRE!!!")
else:
    print("There is a NORMAL....")
