import cv2
import time   
from hand_segmentation import hand_segmentation 
from knn import knn
vid = cv2.VideoCapture(0)
  
while(True):
      
    ret, frame = vid.read()
  
    cv2.imshow('frame', frame)
    hand = hand_segmentation(frame,25) 

    print(hand.shape)
    
    try:
        cv2.imshow('hand_segmentation', hand)
    except:
        print("lost hand")
    
    edges = cv2.Canny(hand, 50, 100)
    try:
        cv2.imshow("edges",edges)
    except:
        print("lost hand")
    
    q = knn(edges)
    img = cv2.imread(q)
    cv2.imshow('found image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
