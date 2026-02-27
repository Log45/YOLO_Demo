from ultralytics import YOLO

pose = YOLO('yolo11n-pose.pt', "pose")
pose.export(format="onnx", dynamic=True, simplify=True)

detect = YOLO('yolov8n.pt')
detect.export(format="onnx", dynamic=True, simplify=True)