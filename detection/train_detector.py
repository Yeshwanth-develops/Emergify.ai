from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="datasets/detection/emergency.yaml",
    epochs=3,
    imgsz=416,
    batch=16,
    device="cpu",
    workers=0,

    augment=False,
    mosaic=0.0,
    copy_paste=0.0,
    mixup=0.0,

    project="detection",
    name="emergency_yolo_fast"
)

print("✅ YOLO training completed")
