### Experiments on TAO Dataset


#### 1. Prepare the data

Please refer to [GET_STARTED](GET_STARTED.md) to prepare the TAO dataset. 

Remember to rename the folder name of `annotations_**` to `annotations`.

#### 2. Generate our annotation files

a. Generate TAO annotation files with 482 classes.
```shell
python tools/convert_datasets/tao2coco.py -t ./data/tao/annotations --filter-classes
```

b. Merge LVIS and COCO datasets to train detectors.

Use the `merge_coco_with_lvis.py` script in [the offical github](https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/merge_coco_with_lvis.py).

This operation follows the practices in [TAO](https://taodataset.org/) with LVIS v0.5.

```shell
cd ${TAP_API}
python ./scripts/detectors/merge_coco_with_lvis.py --lvis ${LVIS_PATH}/annotations/lvis_v0.5_train.json --coco ${COCO_PATH}/annotations/instances_train2017.json --mapping data/coco_to_lvis_synset.json --output-json ${LVIS_PATH}/annotations/lvisv0.5+coco_train.json
```

You can also get the merged annotation file from [here]().

#### 3. Pre-train the model on LVIS dataset

