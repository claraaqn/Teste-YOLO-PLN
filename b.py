import cv2
from ultralytics import YOLO  # Supondo que você esteja usando a implementação do YOLOv5
import math

# Load a model (substitua 'yolov8n.pt' pelo caminho do seu modelo)
model = YOLO('yolov8n.pt')  

# Iniciar captura de vídeo da câmera
cap = cv2.VideoCapture(0)  # 0 indica a câmera padrão
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# object classes
classNames = model.model.names


while True:
    # Capturar frame da câmera
    ret, img = cap.read()
    results = model(img, stream=True)
    
    
    # coordinates
    for r in results.xyxy:
        for *box, conf, cls in r:
            # Coordenadas do retângulo delimitador
            x1, y1, x2, y2 = map(int, box)

            # Desenhar o retângulo delimitador na imagem
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Exibir a classe e a confiança
            cv2.putText(img, f'{classNames[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Webcam', img)

    # Sair pressionando 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()