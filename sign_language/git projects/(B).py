import cv2
import mediapipe as mp

# Initialize the MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the drawing module for visualization
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam or video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand keypoints
    results = hands.process(frame_rgb)

    # Initialize a list to store hand landmarks
    lm_list = []

    # If hands are detected, draw the keypoints and analyze them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract hand landmarks and add them to lm_list
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Define conditions to detect the "B" gesture
            # Check if the thumb is extended, the index finger is bent, and the other fingers are closed
            if lm_list[4][1] < lm_list[5][1] and lm_list[8][1] > lm_list[6][1] and lm_list[12][1] > lm_list[10][1] and lm_list[16][1] > lm_list[14][1] and lm_list[20][1] > lm_list[18][1]:
                cv2.putText(frame, "B", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("B")

    # Display the frame with keypoints
    cv2.imshow('Hand Keypoints', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
