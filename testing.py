from ultralytics import YOLO
import cv2

def process_video(model_path, video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = 'output.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    CONFIDENCE_THRESHOLD = 0.60

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)[0]
            for box in results.boxes:
                if box.conf[0] >= CONFIDENCE_THRESHOLD:
                    # Extrair coordenadas e converter para inteiros
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = f"{model.names[class_id]} {conf:.2f}"

                    # Desenhar retângulo e texto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow("YOLOv11n", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

model_path = "/home/eduardo/Desktop/yolo/fine-tuning-drones/runs/detect/train3/weights/best.pt"
video_path = "/home/eduardo/Desktop/yolo/fine-tuning-drones/Screencast from 2025-05-13 21-12-01.mp4"
process_video(model_path, video_path)
