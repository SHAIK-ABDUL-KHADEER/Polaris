import cv2
import time
import mediapipe as mp
from djitellopy import Tello

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# Initialize Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Gesture control variables
last_gesture = None
last_time = 0
gesture_delay = 2  # seconds
is_flying = False
last_hand_time = time.time()
hand_timeout = 5  # seconds

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index to pinky
    count = 0

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        count += 1

    # Fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    return count

def perform_gesture_action(finger_count):
    global last_gesture, last_time, is_flying

    current_time = time.time()
    if last_gesture == finger_count and (current_time - last_time) < gesture_delay:
        return

    last_gesture = finger_count
    last_time = current_time

    print(f"Detected {finger_count} fingers")

    try:
        if finger_count == 1 and not is_flying:
            print("Takeoff")
            tello.takeoff()
            is_flying = True
            time.sleep(2)  # allow drone to stabilize

        elif finger_count == 2 and is_flying:
            print("Moving up")
            tello.move_up(40)

        elif finger_count == 3 and is_flying:
            print("Moving down")
            tello.move_down(30)

        elif finger_count == 4 and is_flying:
            print("Preparing to backflip...")
            time.sleep(1)  # slight delay before flipping
            tello.flip_back()
            print("Flip done!")

        elif finger_count == 5 and is_flying:
            print("Landing")
            tello.land()
            is_flying = False

    except Exception as e:
        print(f"[ERROR] {e}")

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(hand_landmarks)
                perform_gesture_action(fingers)
                last_hand_time = current_time
        else:
            if is_flying and (current_time - last_hand_time > hand_timeout):
                print("No hand detected for 5 seconds! Emergency landing.")
                try:
                    tello.land()
                except:
                    pass
                is_flying = False

        cv2.imshow("Tello Hand Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if is_flying:
        print("Landing before exit...")
        try:
            tello.land()
        except:
            pass
    tello.end()
