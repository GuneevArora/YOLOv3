"""Constants taken from the YOLOv3 paper and the reference Darknet config (yolov3.cfg)."""

# Section 2.3: nine k-means priors on COCO (width, height) in pixels at 416x416,
# sorted by area and split evenly across the three scales.
ANCHORS = (
    (10, 13), (16, 30), (33, 23),       # stride 8  (small objects)
    (30, 61), (62, 45), (59, 119),      # stride 16 (medium objects)
    (116, 90), (156, 198), (373, 326),  # stride 32 (large objects)
)
# Output order follows Darknet: the coarsest scale is predicted first.
ANCHOR_MASKS = ((6, 7, 8), (3, 4, 5), (0, 1, 2))
STRIDES = (32, 16, 8)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Training hyper-parameters from yolov3.cfg.
TRAIN = dict(
    batch_size=64,
    subdivisions=16,
    learning_rate=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
    burn_in=1000,
    max_batches=500200,
    steps=(400000, 450000),
    scales=(0.1, 0.1),
    jitter=0.3,
    hue=0.1,
    saturation=1.5,
    exposure=1.5,
    multiscale=(320, 608),   # "random=1": new size every 10 batches, multiples of 32
    multiscale_interval=10,
)
