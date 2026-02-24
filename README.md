# League of Legends - detection exploration
Project to capture Leauge of Legends elements using detection, classification and segmentation. 

Data for models are provided via outplay software and my own gameplay.
Data was annotated with CVAT (currently ~5000 objects). 

Detection:
Champion,
Allied minion,
Enemy minion

Classification:
Alive,
Dead,
Lane,
Jungle,
Shop

Segmentation:
Large Raptor,
Small Raptors,
Large Wolf,
Small Wolves,
Large Krug,
Medium Krug,
Small Krugs,
Gromp,
Blue Sentinel,
Red Brambleback,
Scuttle Crab

Models:
Yolov11s/m
Yolov26s/m
RF-DETR
Seg-former


### Example - Yolov11s
![gif_yolo](https://github.com/user-attachments/assets/7fd671ae-6501-45aa-a081-73e4da134b11)


Main goal of project is to collect data with ML techniques from whole Leauge of Legends matches.
