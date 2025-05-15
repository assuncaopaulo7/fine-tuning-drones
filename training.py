from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    print("GPU Available: ", torch.cuda.is_available())
    print("GPU Name: ", torch.cuda.get_device_name(0))

    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(
        data="/home/eduardo/Desktop/yolo/fine-tuning-drones/UAVs-2/data.yaml", 
        epochs=200, 
        imgsz=640,
        device=0,
        name="train"
        )
    
    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("/home/eduardo/Desktop/yolo/fine-tuning-drones/drone-classification-1/test/images/8093a000-7742-11ef-8df0-fdff75f5dac1_jpg_webp.rf.e50da7ab6dda6736f4386fafef53c807.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model