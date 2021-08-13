## Experiments on MOT17 Dataset

### 1. Download and prepare the data

Please download the data from MOT Challenge.

It is recommended to symlink the dataset root to `$QDTrack/data`.

If your folder structure is different, you may need to change the corresponding paths in config files.

Our folder structure follows

```
├── qdtrack
├── tools
├── configs
├── data
    ├── MOT17
        ├── train
        ├── test
```


### 2. Generate our annotation files

```shell
python tools/convert_datasets/mot2coco.py -i ./data/MOT17 -o data/MOT17/annotations --split-train --convert-det
```

### 3. Pre-train the model on MS COCO



### 4. Fine-tune the model on MOT17


