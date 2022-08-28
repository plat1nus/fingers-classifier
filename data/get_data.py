import os
import cv2

VIDEOS_PATH = "/home/plat1nus/Документы/Platinus-fingers-recognition/data/videos"

DATA_PATH = "/home/plat1nus/Документы/Platinus-fingers-recognition/data/train_data"

classes = (0, 1, 2, 3, 4, 5)

os.chdir(VIDEOS_PATH)

for i, video in enumerate(sorted(os.listdir())):
    print(video)
    video_path = os.path.join(VIDEOS_PATH, video)
    cap = cv2.VideoCapture(video_path)
    for j in range(200):
        scs, image = cap.read()
        image = cv2.flip(image,-1)
        image = cv2.resize(image, (224, 224))
        cv2.imshow("image", image)
        cv2.imwrite(f"/home/plat1nus/Документы/Platinus-fingers-recognition/data/train_data/{i}/{j + 1}.jpg", image)
        cv2.waitKey(1)
