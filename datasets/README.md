# Prepare Datasets for D2Zero

### Expected dataset structure:

```
coco
├── train2014
    ├── COCO_train2014_xxxxxxxxxxxx.jpg
    └── ...
├── val2014
    ├── COCO_val2014_xxxxxxxxxxxx.jpg
    └── ...
└── annotations
    └── ZSIS
        ├── instances_train2014_seen_48_17.json
        ├── instances_train2014_seen_65_15.json
        ├── instances_val2014_gzsi_48_17.json
        ├── instances_val2014_gzsi_65_15.json
        ├── instances_val2014_unseen_48_17.json
        └── instances_val2014_unseen_65_15.json

```

Please download the annotations [here](https://drive.google.com/drive/folders/1TLbmDoRiKcMGq1zyVahXtGVTdkvI9Dus) and follow the setting of the [ZSI](https://github.com/zhengye1995/Zero-shot-Instance-Segmentation).



