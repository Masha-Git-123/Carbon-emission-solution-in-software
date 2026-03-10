import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the AI Model
np.set_printoptions(suppress=True)
print("Loading AI model... Please wait.")
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# 2. Start the Webcam
cap = cv2.VideoCapture(0)
print("Webcam started! Press 'q' on your keyboard to close the window.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        break

    # 3. Prepare the image for the AI
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_image_array = (image_array / 127.5) - 1

    # 4. Get the AI's prediction
    prediction = model.predict(normalized_image_array, verbose=0)
    
    # Get the winning index (0 = First Class, 1 = Second Class, 2 = Third Class)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print to terminal
    print(f"Seeing: {class_name} | Confidence: {confidence_score * 100:.1f}%")

    # 5. Draw the symbols based on the INDEX number
    if confidence_score > 0.60:
        
        if index == 0:  
            # 0 is the First Class (GOOD PIN) -> Draw Green Tick ✅
            cv2.line(frame, (50, 100), (80, 130), (0, 255, 0), 10)
            cv2.line(frame, (80, 130), (140, 40), (0, 255, 0), 10)
            cv2.putText(frame, "PASS", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
        elif index == 1: 
            # 1 is the Second Class (BAD PIN) -> Draw Red Cross ❌
            cv2.line(frame, (50, 50), (120, 120), (0, 0, 255), 10)
            cv2.line(frame, (120, 50), (50, 120), (0, 0, 255), 10)
            cv2.putText(frame, "FAIL", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 6. Show the video window
    cv2.imshow("Safety Pin Quality Control", frame)

    # 7. Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()