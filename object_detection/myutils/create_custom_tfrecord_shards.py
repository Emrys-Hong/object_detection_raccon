import hashlib
import os
import io
import math
import pdb

from PIL import Image
import pandas as pd
import tensorflow as tf
import sys
sys.path.append('..')
from object_detection.utils import dataset_util

flags = tf.app.flags

# flags.DEFINE_string('anno_train_file_name', None, 'Annotation training csv file')
# flags.DEFINE_string('anno_val_file_name', None, 'Annotation validation csv file')
flags.DEFINE_string('annotation_file_name', None, 'file path of annotation csv')
flags.DEFINE_string('train_split', 0.8, 'Value between [0,1]')
flags.DEFINE_string('label_map_path', '.', 'path to output .pbtxt category-index map file')
flags.DEFINE_string('output_tfrecord_dir', '.', 'output tfrecord directory')
flags.DEFINE_string('data_dir', None, 'root directory of data')
flags.DEFINE_integer('NUM_SHARDS', 1, 'Number of shards')

FLAGS = flags.FLAGS

def _get_dataset_filename(dataset_dir, split_name, shard_id, _NUM_SHARDS):
  output_filename = '%s_%05d-of-%05d.record' % (
      split_name, shard_id+1, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def create_label_map(category_list, map_path):
    """Create category-index map file"""
    label_map_dict = {}
    with open(map_path + '/label_map.pbtxt', 'w') as f:
        count = 1
        for label in category_list:
            f.write('item {\n')
            f.write('\tid: ' + str(count) + '\n')
            f.write('\tname: ' + label + '\n')
            f.write('}\n')
            label_map_dict[label] = count
            count = count + 1

    return label_map_dict


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=True,
                       image_subdirectory='JPEGImages'):
    full_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    #full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    #truncated = []
    #poses = []
    #difficult_obj = []
    for obj in data['object']:
        #difficult = bool(int(obj['difficult']))
        #if ignore_difficult_instances and difficult:
        #    continue

        #difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        #truncated.append(int(obj['truncated']))
        #poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        #'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        #'image/object/truncated': dataset_util.int64_list_feature(truncated),
        #'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main():
    """Main Function"""

    # assert FLAGS.train_split, 'train_split information is missing'
    assert FLAGS.annotation_file_name, 'file path of annotation csv'
    #assert FLAGS.label_map_path, 'output dir for label_map file missing'
    #assert FLAGS.output_tfrecord_dir, 'output path for tfrecord directory missing'
    assert FLAGS.data_dir, 'root directory of data missing'

    if not os.path.exists(FLAGS.annotation_file_name):
        print("annotation file name does not exist")
        return -1
    if not os.path.exists(FLAGS.label_map_path):
        print("config path does not exist")
        return -1
    if not os.path.exists(FLAGS.output_tfrecord_dir):
        print("the output dir for writing tfrecord does not exist")
        return -1
    if not os.path.exists(FLAGS.data_dir):
        print("the root directory for data does not exist")

    data_dir = FLAGS.data_dir

    train_output_filenames = []
    for shard_id in range(FLAGS.NUM_SHARDS):
        train_output_filenames.append(_get_dataset_filename(FLAGS.output_tfrecord_dir, 'train', shard_id, FLAGS.NUM_SHARDS))

    eval_output_filenames = []
    for shard_id in range(FLAGS.NUM_SHARDS):
        eval_output_filenames.append(_get_dataset_filename(FLAGS.output_tfrecord_dir, 'eval', shard_id, FLAGS.NUM_SHARDS))


    df = pd.read_csv(FLAGS.annotation_file_name, sep=',',
                     names=["name", "height", "width", "label", "xmin", "ymin", "xmax", "ymax"])

    # df.reindex(np.random.permutation(df.index))
    # df = df.sample(frac=1).reset_index(drop=True)

    filenames = df.name.unique()
    total_count = len(filenames)
    label_map_dict = create_label_map(df.label.unique(), FLAGS.label_map_path)

    groupby_image = df.groupby(df['name'])
    train_num_per_shard = int(math.ceil(FLAGS.train_split * total_count / float(FLAGS.NUM_SHARDS)))
    eval_num_per_shard = int(math.ceil((total_count - (FLAGS.train_split * total_count)) / float(FLAGS.NUM_SHARDS)))


    # shuffle the examples by name
    # groupby_image.reindex(np.random.permutation(groupby_image.index))

    # write train.tfrecord

    train_shard_id = 0
    #print(train_shard_id)
    train_start_ndx = train_shard_id * train_num_per_shard
    train_end_ndx = min((train_shard_id+1) * train_num_per_shard, FLAGS.train_split * total_count)
    #print(str(train_start_ndx),',',str(train_end_ndx))
    writer_train = tf.python_io.TFRecordWriter(train_output_filenames[train_shard_id])

    eval_shard_id = 0
    print(eval_shard_id)
    eval_start_ndx = eval_shard_id * eval_num_per_shard
    eval_end_ndx = min((eval_shard_id+1) * eval_num_per_shard, (total_count - (FLAGS.train_split * total_count)))
    print(str(eval_start_ndx),',',str(eval_end_ndx))
    writer_eval = tf.python_io.TFRecordWriter(eval_output_filenames[eval_shard_id])

    train_count = 0
    eval_count = 0
    for name, group in groupby_image:
        #print(str(train_count)+" , "+str(eval_count))
        data = {}
        data['folder'] = data_dir
        data['filename'] = name + '.jpg'
        data['size'] = {}
        data['object'] = []
        for index, row in group.iterrows():
            obj = {}
            obj['bndbox'] = {}
            obj['bndbox']['xmin'] = row['xmin']
            obj['bndbox']['xmax'] = row['xmax']
            obj['bndbox']['ymin'] = row['ymin']
            obj['bndbox']['ymax'] = row['ymax']
            data['size']['width'] = row['width']
            data['size']['height'] = row['height']
            obj['name'] = row['label']
            data['object'].append(obj)

        example = dict_to_tf_example(data, data_dir, label_map_dict)
        if train_count < FLAGS.train_split * total_count:
            writer_train.write(example.SerializeToString())
            train_count += 1
            if (train_count > train_end_ndx) and (train_shard_id < FLAGS.NUM_SHARDS):
                writer_train.close()
                train_shard_id = train_shard_id + 1
                #print(train_shard_id)
                train_start_ndx = train_shard_id * train_num_per_shard
                train_end_ndx = min((train_shard_id+1) * train_num_per_shard, FLAGS.train_split * total_count)
                #print(str(train_start_ndx),',',str(train_end_ndx))
                writer_train = tf.python_io.TFRecordWriter(train_output_filenames[train_shard_id])
        else:
            writer_eval.write(example.SerializeToString())
            eval_count += 1
            if (eval_count > eval_end_ndx) and (eval_shard_id < FLAGS.NUM_SHARDS):
                writer_eval.close()   
                eval_shard_id = eval_shard_id + 1
                #print(eval_shard_id)
                eval_start_ndx = eval_shard_id * eval_num_per_shard
                eval_end_ndx = min((eval_shard_id+1) * eval_num_per_shard, (total_count - (FLAGS.train_split * total_count)))
                #print(str(eval_start_ndx),',',str(eval_end_ndx))
                writer_eval = tf.python_io.TFRecordWriter(eval_output_filenames[eval_shard_id])

    writer_train.close()
    writer_eval.close()


if __name__ == '__main__':
    main()
