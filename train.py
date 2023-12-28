from ultralytics import YOLO

model = YOLO("best.pt")   # the best.pt file obtained after training

model.train(data = "data_custom.yaml", batch=8,imgsz=640, epochs=100, pothole=1)

# "data_custom.yaml" is the file containing the dataset paths information and the number
# objects along with the name of the object.