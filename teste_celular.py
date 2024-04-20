import cv2
from ultralytics import YOLO  # Supondo que você esteja usando a implementação do YOLOv5
import math

# Load a model (substitua 'yolov8n.pt' pelo caminho do seu modelo)
model = YOLO('yolov8n.pt')  

stream = 'http://192.168.10.7:4747/video'
# Iniciar captura de vídeo da câmera
cap = cv2.VideoCapture(stream)  # 0 indica a câmera padrão
cap.set(3, 640)
cap.set(4, 480)

# object classes
classNames = model.model.names


while True:
    # Capturar frame da câmera
    ret, img = cap.read()
    results = model(img, stream=True)
    
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)

    # Sair pressionando 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()