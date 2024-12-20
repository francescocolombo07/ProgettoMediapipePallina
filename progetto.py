import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Configurazione della finestra
screen_width = 640
screen_height = 480
ball_radius = 15

# Impostazioni pallina rossa
ball_x = screen_width // 2
ball_y = screen_height // 2

# rilevamento della mano
hand_open = True  # mano inizialmente aperta
holding_ball = False  # Indica se la pallina è controllata dalla mano
palm_center = (0, 0)  # Centro del palmo
palm_radius = 30  # Raggio per il perimetro del palmo

# Avvia la videocamera
cap = cv2.VideoCapture(0)
frame_witdh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"altezza: {frame_height}, larghezza: {frame_witdh}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # effetto specchio
    frame = cv2.flip(frame, 1)

    # immagine in RGB per MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva mani
    results = hands.process(rgb_frame)

    # Dati rilevamento  mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Disegna landmarks della mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Identifica se mano è aperta o chiusa
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

            if distance > 0.2:  # Soglia per determinare se la mano è aperta
                hand_open = True
            else:
                hand_open = False

            # Calcola centro del palmo
            palm_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * screen_width)
            palm_y = int((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y +
                          hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y) /2 * screen_height)
            palm_center = (palm_x, palm_y)

    # Gestione della pallina
    if not holding_ball:  # Se la pallina non è ancora in controllo
        # Controlla se pallina è nel perimetro del palmo
        if np.sqrt((palm_center[0] - ball_x)**2 + (palm_center[1] - ball_y)**2) < palm_radius:
            if not hand_open:  # Se la mano è chiusa
                holding_ball = True  # La pallina è ora in controllo della mano

    if holding_ball:
        # pallina segue il centro del palmo
        ball_x, ball_y = palm_center

    if holding_ball and hand_open:  # Se la mano si apre, rilascia la pallina
        holding_ball = False

    # Disegna la pallina rossa
    cv2.circle(frame, (int(ball_x), int(ball_y)), ball_radius, (0, 0, 255), -1)

    # Disegna il perimetro del palmo
    cv2.circle(frame, palm_center, palm_radius, (0, 255, 0), 2)

    cv2.imshow('Gioco Pallina', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()