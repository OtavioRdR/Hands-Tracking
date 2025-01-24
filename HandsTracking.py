import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
points = []

def indicador_levantado(hand_landmarks):
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        return True
    return False

def medio_levantado(hand_landmarks):
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        return True
    return False

def apagar_linha():
    global points
    points = []

while True:
    success, image = cap.read()
    if not success:
        print("Falha ao capturar imagem da câmera.")
        continue
    
    image = cv2.flip(image, 1)
    imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLms, mphands.HAND_CONNECTIONS)
            if indicador_levantado(handLms):
                h, w, _ = image.shape
                cx, cy = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
                points.append((cx, cy))
            if indicador_levantado(handLms) and medio_levantado(handLms):
                apagar_linha()
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(image, points[i - 1], points[i], (0, 0, 255), 5)

    cTime = time.time()
    fps = round(1 / (cTime - pTime), 2)
    pTime = cTime
    cv2.putText(image, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("imagem", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
