from ultralytics import YOLO

model_cfg = "yolov8m_p2_se.yaml"
data_cfg  = "~/manipulator_project_datasets/data.yaml"
device    = 0

train_args = dict(
    imgsz          = 640,
    epochs         = 130,
    batch          = 24,
    workers        = 8,
    device         = device,
    optimizer      = "SGD",
    lr0            = 0.005,
    lrf            = 0.01,
    momentum       = 0.9,
    weight_decay   = 5e-4,
    mosaic         = 1.0,
    mixup          = 0.2,
    hsv_h          = 0.015,
    hsv_s          = 0.7,
    hsv_v          = 0.4,
    flipud         = 0.0,
    fliplr         = 0.5,
    scale          = 0.7,
    degrees        = 0.0,
    translate      = 0.1,
    patience       = 20,
    save_period    = 10,
    seed           = 0,
    amp            = True,
    project        = 'runs/train',
    name           = 'custom_yolo_experiment',
    save           = True,
    plots          = True,
    val            = True,
)

if __name__ == "__main__":
    model = YOLO(model_cfg)
    model.train(data=data_cfg, **train_args)

