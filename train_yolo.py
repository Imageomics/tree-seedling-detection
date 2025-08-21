import comet_ml
import ultralytics
from ultralytics import YOLO
import argparse

comet_ml.login(project_name="forest-yolov11-rgb")


# Load YOLOv11 model (start with pretrained weights for faster convergence)
model = YOLO("yolo11n.pt")   # you can also try yolov11s.pt, yolov11m.pt etc.

project_name = "forest-yolov11-rgb"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default='rgb')
    parser.add_argument("--epoch", type=int, default=100)
    
    args = parser.parse_args()
    
    parameters = {
    "epochs": args.epoch,
    "batch_size": args.batch_size,
    "modality":args.modality
}

    data_yaml_path = "/blue/azare/zhou.m/funca_2025/forest/FOREST/forest_yolo_dataset/data.yaml"

    # train the model
    results = model.train(
        data=data_yaml_path,
        epochs=parameters['epochs'],
        patience = int(parameters['epochs']*0.2),
        batch=parameters['batch_size'],
        save_period=int(parameters['epochs']*0.5),
        device='0',
        project=project_name,
        verbose=False,
        val=True,
        seed=42)



    # Evaluate on the test set (YOLO calls it 'val')
    metrics = model.val()

    # Print summary metrics
    print("✅ Training complete.")
    print("Test Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")      # mAP at IoU 0.5
    print(f"mAP50-95: {metrics.box.map:.4f}")     # mAP averaged over IoU 0.5-0.95
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")