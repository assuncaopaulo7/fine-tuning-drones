from roboflow import Roboflow
rf = Roboflow(api_key="Xoh7u7gqGl9rPaafwW7i")
project = rf.workspace("assuncaopaulo7").project("drone-classification-23pmh")
version = project.version(1)
dataset = version.download("yolov11") 
                