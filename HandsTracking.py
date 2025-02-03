import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

def indicador_levantado(hand_landmarks):
    """Retorna True se o dedo indicador estiver levantado"""
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    object_boxes = []  

 
    yolo_results = model(image, conf=0.5, verbose=False)
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])] if box.cls[0] in model.names else "Desconhecido"
            object_boxes.append((x1, y1, x2, y2, label))

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = image.shape
            cx, cy = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
            
            for (x1, y1, x2, y2, label) in object_boxes:
                if x1 < cx < x2 and y1 < cy < y2:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Detecção", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
