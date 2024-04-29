import cv2  # Install opencv-python
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
import serial  # Import serial library for communication

# Define serial port and baud rate (adjust these if needed)
ser = serial.Serial('COM16', 9600)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window with class and confidence score overlay
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Prepare class and confidence score strings for display
    cls_str = "Class: " + str(class_name[2:]) + "\n"
    conf_str = "Confidence Score: " + str(np.round(confidence_score * 100))[:-2] + "%"
    status = cls_str + conf_str

    # Print prediction and confidence score
    print(status)

    # Display class and confidence score on the image
    color = (255, 0, 0)
    cv2.putText(image, status, (1, 1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Check if detected class is "pos" and send signal if so
    if class_name == "pos":
        print("Target Detected! Triggering Servo Movement...")
        ser.write(b't')  # Send 'S' signal to Arduino

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
