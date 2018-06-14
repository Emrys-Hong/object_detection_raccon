# fs-object-detection

This repo has a skimmed down version of the Tensorflow Object Detection API with
training and evaluation as a single pipeline.

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:


``` bash
# From fs-object-detection/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the fs-object-detection/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From fs-object-detection/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file.

# Organization of Data

The image data needs to be organized in the following folder hierarchy

```
+ImageData
 +JPEGImages
  -*.jpg
 ```

## Create train/eval TFRecord

The TFRecord can be generated from annotation folder xml format in PascalVOC format.
It can also be generated from a single annotation csv file. This is for customers who 
do not want to deal with XML files. The annotations need to be in the following format:

```
imageName,height,width,category,xmin,ymin,xmax,ymax ## but here you need to delete the first line
2007_000027,500,486,person,174,101,349,351
```
Once you have a csv file in the above format the following code can be used to generate
the TFRecord file. Before that create the following directories ```tfrecord_data```, ```train_output```, ```eval_output``` under the ```<ROOT_FOLDER_OF_CUSTOMER_DIR>``` folder.

```
python object_detection/myutils/create_custom_tfrecord.py \
    --annotation_file_name=<ROOT_FOLDER_OF_CUSTOMER_DIR>/annotation.csv \
    --train_split=0.8 \
    --label_map_path=<ROOT_FOLDER_OF_CUSTOMER_DIR> \
    --data_dir=<ROOT_FOLDER_OF_CUSTOMER_DIR> \
    --output_tfrecord_dir=<ROOT_FOLDER_OF_CUSTOMER_DIR>/tfrecord_data
```
This will generate 2 files in the output_tfrecord_dir folder: ```train.record``` and ```eval.record```
corresponding to the train and evaluation data. The default train-eval split is 80% but this 
can be changed as shown above.

It also generates the label_map file ```label_map.pbtxt``` in the ```label_map_path``` directory that 
contains the list of categories in the dataset. This will be used later.

Note that depending on the size of the dataset the train and eval files can be quite big leading to
to longer training and evaluation times.

To make the training and evaluation run faster you can actually break your training and evaluation
data into shards which makes the training and evaluation faster. This is primarily to
assists faster experimentation. For generating TFRecord shards do the following

```
python object_detection/myutils/create_custom_tfrecord_shards.py \
    --annotation_file_name=<ROOT_FOLDER_OF_CUSTOMER_DIR>/annotation.csv \
    --train_split=0.8 \
    --label_map_path=<ROOT_FOLDER_OF_CUSTOMER_DIR> \
    --data_dir=<ROOT_FOLDER_OF_CUSTOMER_DIR> \
    --output_tfrecord_dir=<ROOT_FOLDER_OF_CUSTOMER_DIR>/tfrecord_data \
    --NUM_SHARDS=3
```
This will generate 3 shards for training and evaluation in the output_tfrecord_dir folder:
```train_00001-of-00003.record```, ```train_00002-of-00003.record```, ```train_00003-of-00003.record```,
```eval_00001-of-00003.record```, ```eval_00002-of-00003.record```, ```eval_00003-of-00003.record```

Any pair of the above shards can be used for training and evaluation.

## Generate train/eval config file

The next step is to generate the training/evaluation config file that will be used by the training
and evaluation stages. The config provided is for fasterRCNN architecture which takes an image and 
outputs bounding box coordinates, bounding box category and bounding box scores. The base architecture
used is ResNet101 for generating the ROI proposals.

```
python object_detection/myutils/genconfig_faster_rcnn_resnet101.py \
    --pre_trained_model_path=faster_rcnn_resnet101_coco_pre-trained_model \
    --train_tfrecord_path=<ROOT_FOLDER_OF_CUSTOMER_DIR>/tfrecord_data \
    --val_tfrecord_path=<ROOT_FOLDER_OF_CUSTOMER_DIR>/tfrecord_data \
    --label_map_path=<ROOT_FOLDER_OF_CUSTOMER_DIR>/label_map.pbtxt \
    --config_path=<ROOT_FOLDER_OF_CUSTOMER_DIR> \
    --num_steps=xx \
    --dataset_name=${ANY_NAME}
```
Note that you need to have your pre-trained model in the ```faster_rcnn_resnet101_coco_pre-trained_model``` directory.
An example pre-trained model (faster_rcnn_resnet101_coco_2017_11_08) downloaded from the Tensorflow 
Object Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) which was trained on the MSCOCO dataset. The ```num_steps``` is a variable and can be set depending on the size of the dataset and the number of categories. Default value is ```200000``` (based on a dataset of size 10000 images and 20 categories). 

```
+faster_rcnn_resnet101_coco_pre-trained_model
 -checkpoint
 -model.ckpt.data-00000-of-00001
 -model.ckpt.index
 -model.ckpt.meta
```


## Training

```
# From the fs-object-detection/ directory
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${FULL_PATH_TO_CONFIG_FILE} \
    --train_dir=${PATH_TO_OUTPUT_TRAIN_MODELS_EVENTS}
```
You may need to create the ```${PATH_TO_OUTPUT_TRAIN_MODELS_EVENTS}``` directory. Here ```${FULL_PATH_TO_CONFIG_FILE}``` is ```<ROOT_FOLDER_OF_CUSTOMER_DIR>/faster_rcnn_resnet101_${dataset_name}.config```. 

## Evaluation

```
# From the fs-object-detection/ directory
python object_detection/eval.py \
    --logtostderr \
    --report_filename=<ROOT_FOLDER_OF_CUSTOMER_DIR>/${EVALUATION_REPORT_FILE_NAME} \
    --pipeline_config_path=${FULL_PATH_TO_CONFIG_FILE} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
```
Here ```eval_dir``` is the directory where the output of the evaluatuon is written including the events summary. ```checkpoint_dir``` is the ```train_dir``` where checkpoint files are stored regularly as out of the training step. The ```report_filename``` is the filename that contains the evaluation report (basically mAP for each category at each iteration). The output of evaluation is also the best model out of all those models and is denoted by the name ```best_report_filename``` in the ```<ROOT_FOLDER_OF_CUSTOMER_DIR>``` directory.

## Export Inference Graph

```
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG}\
    --trained_checkpoint_prefix ${PATH_TO_OUTPUT_TRAIN_MODELS_EVENTS}/model.ckpt-XXXXX \
    --output_directory ${PATH_TO_SAVED_MODEL} \
    --export_as_saved_model True
```
Here XXXXX is the iteration number of the model that gave the best learned model. The number XXXXX can be found in the file best*.csv in the home directory of the ```<ROOT_FOLDER_OF_CUSTOMER_DIR>``` directory and is the iteration number of the best model.
