# facial-emotion-detection

The provided Python code is a real-time facial emotion detection program that uses a pre-trained convolutional neural network (CNN) model. The program captures a video stream from the default camera and processes each frame by detecting faces and predicting the emotion associated with each face. Here is a detailed summary of the code:

1. The code imports the necessary libraries - cv2 for image processing, numpy for numerical operations and load_model from keras.models for loading the pre-trained CNN model.

2. The pre-trained model is loaded from the saved file "fer2013_mini_XCEPTION.102-0.66.hdf5".

3. The EMOTIONS variable is defined as a list of strings containing the possible emotions that the model can predict.

4. The code initializes the default camera using the cv2.VideoCapture(0) function.

5. The program enters an infinite loop where each iteration captures a frame from the camera using cap.read().

6. The captured frame is then converted to grayscale using cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).

7. The cv2.CascadeClassifier object is used to detect faces in the grayscale image. The Haar cascade classifier haarcascade_frontalface_default.xml is used for face detection.

8. The detectMultiScale() function is called with parameters for scale factor, minimum neighbors, and minimum size to detect the faces in the frame.

9. For each detected face, the program extracts the face region of interest (ROI) from the grayscale image using the numpy slicing operation.

10. The extracted face ROI is then resized to (64,64) pixels and normalized between 0 and 1 using cv2.resize() and astype("float") / 255.0.

11. The normalized face ROI is reshaped into a 4D tensor of shape (1, 64, 64, 1) for input into the CNN model.

12. The CNN model then predicts the probability of each emotion for the face ROI using model.predict(face_roi).

13. The program selects the emotion with the highest probability and displays it as text on the frame using cv2.putText().

14. A green rectangle is drawn around the face ROI using cv2.rectangle().

15. The program displays the resulting frame with emotion text and rectangle using cv2.imshow().

16. The loop continues until the 'q' key is pressed, which breaks out of the loop.

17. The camera is released using cap.release() and all windows are closed using cv2.destroyAllWindows().
