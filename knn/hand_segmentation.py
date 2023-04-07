import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

def hand_segmentation(img, padding):
    #hand class initialization
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

    mp_drawing = mp.solutions.drawing_utils

    # Reading the sample image on which we will perform the detection
    sample_img = cv2.imread('testImg4.jpg')

    # Here we are specifing the size of the figure i.e. 10 -height; 10- width.
    plt.figure(figsize = [10, 10])

    # Here we will display the sample image as the output.
    plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()


    results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    image_height, image_width, _ = sample_img.shape
    img_copy = sample_img.copy()

    jointsArray = np.zeros((21, 3))

    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for i in range(21):
                jointsArray[i, 0] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width
                jointsArray[i, 1] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height
                jointsArray[i, 2] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z * image_width
                
    xMax = int(np.max(jointsArray[:,0]))
    xMin = int(np.min(jointsArray[:,0]))

    yMax = int(np.max(jointsArray[:,1]))
    yMin = int(np.min(jointsArray[:,1]))

    segmmentedHand = img_copy[yMin - padding:yMax + padding, xMin - padding:xMax + padding]

    plt.title("Resultant Image");plt.axis('off');plt.imshow(segmmentedHand[:,:,::-1]);plt.show()


if __name__ == "__main__":
    padding = 150
    img = None
    hand_segmentation(img, padding)
