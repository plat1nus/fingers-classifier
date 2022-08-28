from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
model = VGG19(weights='imagenet')

img_path = 'spider.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

image_output = cv2.imread("spider.jpg")

image_output = cv2.putText(image_output, str(decode_predictions(preds, top=3)[0]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                           (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Image", image_output)
cv2.waitKey(0)

# print(features)
