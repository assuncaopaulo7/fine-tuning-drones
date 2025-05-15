from roboflow import Roboflow
rf = Roboflow(api_key="Xoh7u7gqGl9rPaafwW7i")
project = rf.workspace("uavs-7l7kv").project("uavs-vqpqt")
version = project.version(2)
dataset = version.download("yolov11")
                