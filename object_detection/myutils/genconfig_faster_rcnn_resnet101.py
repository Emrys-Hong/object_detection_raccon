
import tensorflow as tf
import os
import glob

flags = tf.app.flags

flags.DEFINE_string('pre_trained_model_path', None, 'Absolute Path to pre-trained ckpt files (do not add trailing /')
flags.DEFINE_string('train_tfrecord_path', None, 'Absolute Path/train_filename.record')
flags.DEFINE_string('val_tfrecord_path', None, 'Absolute Path/val_filename.record')
flags.DEFINE_string('label_map_path', None, 'Absolute path to .pbtxt category-index map file')
flags.DEFINE_string('config_path', None, 'Absolute output directory for config file')
flags.DEFINE_string('dataset_name', 'pascalvoc', 'Name of the dataset used in the pipeline')

FLAGS = flags.FLAGS

def count_classes(label_map_path):

  count = 0
  with open(label_map_path,'r') as f:
    for line in f:
      c = line[0]
      if c == '}':
        count+=1

  print('Number of classes = '+str(count))
  return count


def set_config(num_classes, pre_trained_model_path, 
  train_tfrecord_path, val_tfrecord_path, label_map_path, config_path):

  if not os.path.exists(config_path):
    print("config path foes not exist")
    return -1
  if not os.path.exists(pre_trained_model_path):
    print("pre-trained ckpts file does not exist")
    return -1
  if not os.path.exists(train_tfrecord_path):
    print("train tfrecord file does not exist")
    return -1
  if not os.path.exists(val_tfrecord_path):
    print("validation tfrecord file does not exist")
    return -1
  if not os.path.exists(label_map_path):
    print("label map path does not exist")
    return -1
  if num_classes <= 0:
    print("Illegal number of classes")
    return -1
  
  # Counting number of training records
  # Commenting as it is not yet required
  #train_count = 0
  #tf_records_filenames = glob.glob(os.path.join(train_tfrecord_path,'train*.record'))
  #for fn in tf_records_filenames:
  #  for record in tf.python_io.tf_record_iterator(fn):
  #      train_count += 1

  #Counting number of validation records
  val_count = 0
  tf_records_filenames = glob.glob(os.path.join(train_tfrecord_path,'val*.record'))
  for fn in tf_records_filenames:
    for record in tf.python_io.tf_record_iterator(fn):
        val_count += 1

  with open(config_path+'/faster_rcnn_resnet101_'+dataset_name+'.config','w') as f:
    f.write('# Faster R-CNN with Resnet-101 (v1), configured for Open Images Dataset.\n')
    f.write('# Users should configure the fine_tune_checkpoint field in the train config as\n')
    f.write('# well as the label_map_path and input_path fields in the train_input_reader and\n')
    f.write('# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that\n')
    f.write('# should be configured.\n\n')
    f.write('model {\n')
    f.write('  faster_rcnn {\n')
    f.write('    num_classes: '+str(num_classes)+'\n')
    f.write('    image_resizer {\n')
    f.write('      keep_aspect_ratio_resizer {\n')
    f.write('        min_dimension: 600\n')
    f.write('        max_dimension: 1024\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('    feature_extractor {\n')
    f.write('      type: \'faster_rcnn_resnet101\'\n')
    f.write('      first_stage_features_stride: 16\n')
    f.write('    }\n')
    f.write('    first_stage_anchor_generator {\n')
    f.write('      grid_anchor_generator {\n')
    f.write('        scales: [0.25, 0.5, 1.0, 2.0]\n')
    f.write('        aspect_ratios: [0.5, 1.0, 2.0]\n')
    f.write('        height_stride: 16\n')
    f.write('        width_stride: 16\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('    first_stage_box_predictor_conv_hyperparams {\n')
    f.write('      op: CONV\n')
    f.write('      regularizer {\n')
    f.write('        l2_regularizer {\n')
    f.write('          weight: 0.0\n')
    f.write('        }\n')
    f.write('      }\n')
    f.write('      initializer {\n')
    f.write('        truncated_normal_initializer {\n')
    f.write('          stddev: 0.01\n')
    f.write('        }\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('    first_stage_nms_score_threshold: 0.0\n')
    f.write('    first_stage_nms_iou_threshold: 0.7\n')
    f.write('    first_stage_max_proposals: 300\n')
    f.write('    first_stage_localization_loss_weight: 2.0\n')
    f.write('    first_stage_objectness_loss_weight: 1.0\n')
    f.write('    initial_crop_size: 14\n')
    f.write('    maxpool_kernel_size: 2\n')
    f.write('    maxpool_stride: 2\n')
    f.write('    second_stage_box_predictor {\n')
    f.write('      mask_rcnn_box_predictor {\n')
    f.write('        use_dropout: false\n')
    f.write('        dropout_keep_probability: 1.0\n')
    f.write('        fc_hyperparams {\n')
    f.write('          op: FC\n')
    f.write('          regularizer {\n')
    f.write('            l2_regularizer {\n')
    f.write('              weight: 0.0\n')
    f.write('            }\n')
    f.write('          }\n')
    f.write('          initializer {\n')
    f.write('            variance_scaling_initializer {\n')
    f.write('              factor: 1.0\n')
    f.write('              uniform: true\n')
    f.write('              mode: FAN_AVG\n')
    f.write('            }\n')
    f.write('          }\n')
    f.write('        }\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('    second_stage_post_processing {\n')
    f.write('      batch_non_max_suppression {\n')
    f.write('        score_threshold: 0.0\n')
    f.write('        iou_threshold: 0.6\n')
    f.write('        max_detections_per_class: 100\n')
    f.write('        max_total_detections: 300\n')
    f.write('      }\n')
    f.write('      score_converter: SOFTMAX\n')
    f.write('    }\n')
    f.write('    second_stage_localization_loss_weight: 2.0\n')
    f.write('    second_stage_classification_loss_weight: 1.0\n')
    f.write('  }\n')
    f.write('}\n\n')

    f.write('train_config: {\n')
    f.write('  batch_size: 1\n')
    f.write('  optimizer {\n')
    f.write('    momentum_optimizer: {\n')
    f.write('      learning_rate: {\n')
    f.write('        manual_step_learning_rate {\n')
    f.write('          initial_learning_rate: 0.0001\n')
    f.write('          schedule {\n')
    f.write('            step: 0\n')
    f.write('            learning_rate: .0001\n')
    f.write('          }\n')
    f.write('          schedule {\n')
    f.write('            step: 500000\n')
    f.write('            learning_rate: .00001\n')
    f.write('          }\n')
    f.write('          schedule {\n')
    f.write('            step: 700000\n')
    f.write('            learning_rate: .000001\n')
    f.write('          }\n')
    f.write('        }\n')
    f.write('      }\n')
    f.write('      momentum_optimizer_value: 0.9\n')
    f.write('    }\n')
    f.write('    use_moving_average: false\n')
    f.write('  }\n')
    f.write('  gradient_clipping_by_norm: 10.0\n')
    f.write('  fine_tune_checkpoint: \"'+pre_trained_model_path+'/model.ckpt\"\n') # Add exception handling here
    f.write('  from_detection_checkpoint: true\n')
    f.write('  num_steps: '+str(200000)+'\n')
    f.write('  data_augmentation_options {\n')
    f.write('    random_horizontal_flip {\n')
    f.write('    }\n')
    f.write('  }\n')
    f.write('}\n')
    f.write('\n')

    f.write('train_input_reader: {\n')
    f.write('  tf_record_input_reader {\n')
    f.write('    input_path: \"'+train_tfrecord_path+'\"\n')
    f.write('  }\n')
    f.write('  label_map_path: \"'+label_map_path+'\"\n')
    f.write('}\n')
    f.write('\n')

    f.write('eval_config: {\n')
    f.write('  num_examples: '+str(val_count)+'\"\n')
    f.write('}\n')
    f.write('\n')

    f.write('eval_input_reader: {\n')
    f.write('  tf_record_input_reader {\n')
    f.write('    input_path: \"'+val_tfrecord_path+'\"\n')
    f.write('  }\n')
    f.write('  label_map_path: \"'+label_map_path+'\"\n')
    f.write('  shuffle: false\n')
    f.write('  num_readers: 1\n')
    f.write('}\n')

    return 1

def main():

  assert FLAGS.pre_trained_model_path, 'pre-trained model path missing'
  assert FLAGS.train_tfrecord_path, 'tfrecord path for train data missing'
  assert FLAGS.val_tfrecord_path, 'tfrecord path for validation data missing'
  assert FLAGS.label_map_path, 'path to .pbtxt category-index map file missing'
  assert FLAGS.config_path, 'dir path to output config file missing'
  assert FLAGS.dataset.name, 'good to give a dataset name to differentiate configs'

  num_classes = count_classes(FLAGS.label_map_path)

  status = set_config(num_classes, FLAGS.pre_trained_model_path, 
    FLAGS.train_tfrecord_path, FLAGS.val_tfrecord_path, 
    FLAGS.label_map_path, FLAGS.config_path)

  if status != 1:
    print("Config file not created ....")
  else:
    print(FLAGS.config_path+'/faster_rcnn_resnet101_'+dataset_name+'.config'+'...created')

if __name__ == '__main__':
  main()


