# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
model=tf.keras.models.load_model("keras/keras_model.h5")
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    image=cv2.resize(frame,(224,224))
    image1=np.array(image,dtype=np.float32)
    image2=np.expand_dims(image1,axis=0)
    image3=image2/255.0
    p=model.predict(image3)
    result=np.argmax(p,axis=1)
    print(result)
    '''if(result==0):
        cv2.putText(frame,"With Hand",(100,100),cv2.FONT_HERSHEY_DUPLEX,2,(234,182,245),2)
    else:
        cv2.putText(frame,"Without Hand" ,(100,100),cv2.FONT_HERSHEY_DUPLEX,2,(234,182,245),2)'''
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()