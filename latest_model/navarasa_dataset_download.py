from roboflow import Roboflow
# Replace with your roboflow API key
rf = Roboflow(api_key="EntfAFHcUMve4NEfGYPt")
# Navarasa dataset info from Roboflow
project = rf.workspace("computervision-yeksm").project("navarasa")
dataset = project.version(1).download("folder") 
