# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import pdb

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to OpenImages dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
#flags.DEFINE_string('year', '2017', 'OpenImages released year.')
flags.DEFINE_string('output_tfrecord_dir', '.', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', None,
                    'Path to output label map proto directory')
#flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
#                                                          'difficult instances')
flags.DEFINE_string('categories', 'person', 'name of category')
FLAGS = flags.FLAGS

SETS = ['train', 'val']
CATEGORIES = ['Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football', 'Ambulance',
    'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ', 'Cassette deck', 'Apple',
    'Eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Beard', 'Bird',
    'Parking meter', 'Traffic light', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll',
    'Skull', 'Washing machine', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart',
    'Ball', 'Backpack', 'Bike', 'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot',
    'Headphones', 'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy', 'Screwdriver', 
    'Bicycle wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill', 'Dress', 'Bear', 'Waffle',
    'Pancake', 'Brown bear', 'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot',
    'Person', 'Bow and arrow', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat', 'Starfish',
    'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Licence plate', 'Lantern',
    'Toaster', 'Flashlight', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 
    'Scissors', 'Stairs', 'Computer keyboard', 'Printer', 'Traffic sign', 'Chair', 'Shirt', 
    'Poster', 'Cheese', 'Sock', 'Fire hydrant', 'Land vehicle', 'Earrings', 'Tie', 
    'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 
    'Clock', 'Medical equipment', 'Cattle', 'Cello', 'Jet ski', 'Camel', 'Coat', 'Suit', 
    'Desk', 'Cat', 'Bronze sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Mouse',
    'Cookie', 'Office', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Computer monitor', 
    'Box', 'Stapler', 'Christmas tree', 'Cowboy hat', 'Hiking equipment', 'Studio couch', 
    'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Mouth', 'Dairy', 'Dice', 
    'Oven', 'Dinosaur', 'Ratchet', 'Couch', 'Cricket ball', 'Winter melon', 'Spatula', 
    'Whiteboard', 'Pencil sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 
    'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Tin can', 'Mug', 'Tap', 'Harbor seal', 
    'Stretcher', 'Can opener', 'Goggles', 'Human body', 'Roller skates', 'Coffee cup', 
    'Cutting board', 'Blender', 'Plumbing fixture', 'Stop sign', 'Office supplies', 
    'Volleyball', 'Vase', 'Slow cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Paper towel', 
    'Personal care', 'Food', 'Sun hat', 'Tree house', 'Flying disc', 'Skirt', 'Gas stove', 
    'Salt and pepper shakers', 'Mechanical fan', 'Face powder', 'Fax', 'Fruit', 'French fries', 
    'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'Horn', 'Window blind', 
    'Foot', 'Golf cart', 'Jacket', 'Egg', 'Street light', 'Guitar', 'Pillow', 'Leg', 'Isopod', 
    'Grape', 'Ear', 'Power plugs and sockets', 'Panda', 'Giraffe', 'Woman', 'Door handle', 
    'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Baseball bat', 
    'Baseball glove', 'Mixing bowl', 'Marine invertebrates', 'Kitchen utensil', 'Light switch', 
    'House', 'Horse', 'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed', 'Adhesive tape', 
    'Harp', 'Sandal', 'Bicycle helmet', 'Saucer', 'Harpsichord', 'Hair', 'Heater', 'Harmonica', 
    'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Drinking straw', 'Insect', 
    'Hair dryer', 'Kitchenware', 'Indoor rower', 'Invertebrate', 'Food processor', 'Bookcase', 
    'Refrigerator', 'Wood-burning stove', 'Punching bag', 'Common fig', 'Cocktail shaker', 
    'Jaguar', 'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet', 'Artichoke', 
    'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Bottle opener', 'Lynx', 
    'Lavender', 'Lighthouse', 'Dumbbell', 'Head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 
    'Billiard table', 'Mammal', 'Mouse', 'Motorcycle', 'Musical instrument', 'Swim cap', 
    'Frying pan', 'Snowplow', 'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Waffle iron', 
    'Milk', 'Ring binder', 'Plate', 'Mobile phone', 'Baked goods', 'Mushroom', 'Crutch', 
    'Pitcher', 'Mirror', 'Lifejacket', 'Table tennis racket', 'Pencil case', 'Musical keyboard', 
    'Scoreboard', 'Briefcase', 'Kitchen knife', 'Nail', 'Tennis ball', 'Plastic bag', 'Oboe', 
    'Chest of drawers', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Hair spray', 
    'Sports equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Infant bed', 'Polar bear', 
    'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Digital clock', 'Pig', 'Reptile', 'Rifle', 
    'Lipstick', 'Skateboard', 'Raven', 'High heels', 'Red panda', 'Rose', 'Rabbit', 'Sculpture', 
    'Saxophone', 'Shotgun', 'Seafood', 'Submarine sandwich', 'Snowboard', 'Sword', 
    'Picture frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 
    'Submarine', 'Scorpion', 'Segway', 'Bench', 'Snake', 'Coffee table', 'Skyscraper', 
    'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 
    'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Picnic basket', 
    'Cooking spray', 'Trousers', 'Bowling equipment', 'Football helmet', 'Truck', 'Measuring cup', 
    'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Paper cutter', 'Wine', 'Weapon', 'Wheel', 
    'Worm', 'Wok', 'Whale', 'Zebra', 'Auto part', 'Jug', 'Pizza cutter', 'Cream', 'Monkey', 
    'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 
    'Crocodile', 'Hippopotamus', 'Toilet', 'Toilet paper', 'Squid', 'Clothing', 'Footwear', 
    'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Wine glass', 'Countertop', 
    'Tablet computer', 'Waste container', 'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark', 
    'Candle', 'Leopard', 'Axe', 'Hand dryer', 'Soap dispenser', 'Porcupine', 'Flower', 
    'Canary', 'Cheetah', 'Palm tree', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 
    'Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 
    'Horizontal bar', 'Convenience store', 'Bench', 'Ice cream', 'Caterpillar', 'Butterfly', 
    'Parachute', 'Orange', 'Antelope', 'Beaker', 'Moths and butterflies', 'Window', 'Closet', 
    'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 
    'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash', 'Racket', 'Face', 'Arm', 
    'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 
    'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 
    'Cake', 'Dragonfly', 'Sunflower', 'Microwave oven', 'Honeycomb', 'Marine mammal', 
    'Sea lion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 
    'Van', 'Grinder', 'Spice rack', 'Light bulb', 'Corded phone', 'Sports uniform', 
    'Tennis racket', 'Wall clock', 'Serving tray', 'Kitchen & dining room table', 'Dog bed', 
    'Cake stand', 'Cat furniture', 'Bathroom accessory', 'Facial tissue holder', 
    'Pressure cooker', 'Kitchen appliance', 'Tire', 'Ruler', 'Luggage and bags', 'Microphone', 
    'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band-aid', 'Animal', 'Bell pepper', 
    'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Nose', 'Pen', 'Ant', 'Car', 
    'Aircraft', 'Hand', 'Skunk', 'Teddy bear', 'Watermelon', 'Cantaloupe', 'Dishwasher', 
    'Flute', 'Balance beam', 'Sandwich', 'Shrimp', 'Sewing machine', 'Binoculars', 
    'Rays and skates', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 
    'Alpaca', 'Taxi', 'Canoe', 'Remote control', 'Wheelchair', 'Rugby ball', 'Armadillo', 
    'Maracas', 'Helmet']
#YEARS = ['2017']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=True,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    full_path = os.path.join(dataset_directory, image_subdirectory, data['filename'])
    print(full_path)
    #full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        #raise ValueError('Image format not JPEG')
        return -1
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

        if obj['name'] in label_map_dict:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            #pdb.set_trace()
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

def create_label_map(category_list, map_path):
    """Create category-index map file"""
    label_map_dict = {}
    with open(map_path + '/label_map.pbtxt', 'w') as f:
        count = 1
        for label in category_list:
            f.write('item {\n')
            f.write('\tid: ' + str(count) + '\n')
            f.write('\tname: \'' + label + '\'\n')
            f.write('}\n\n')
            label_map_dict[label] = count
            count = count + 1

    return

def main():
    pass


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    #if FLAGS.year not in YEARS:
    #    raise ValueError('year must be in : {}'.format(YEARS))
    if not os.path.exists(FLAGS.data_dir):
        print("the root directory for data does not exist")
        return -1
    if not os.path.exists(os.path.join(FLAGS.data_dir,FLAGS.annotations_dir)):
        print("annotation file name does not exist")
        return -1
    if not os.path.exists(FLAGS.label_map_path):
        print("label path does not exist")
        return -1
    if not os.path.exists(FLAGS.output_tfrecord_dir):
        print("the output dir for writing tfrecord does not exist")
        return -1

    data_dir = FLAGS.data_dir
    #years = ['2017']
    #if FLAGS.year != 'merged':
    #    years = [FLAGS.year]
    
    category = FLAGS.categories.split(',')
    for cate in category:
        if cate not in CATEGORIES:
            raise ValueError('category must be in : {}'.format(CATEGORIES))

    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_tfrecord_dir,FLAGS.set+'.record'))

    create_label_map(category, FLAGS.label_map_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path+'/label_map.pbtxt')

    for label in category:
        #logging.info('Reading from OpenImages %s dataset.', year)
        examples_path = os.path.join(data_dir, 'ImageSets', 'Main',
                                     label + '_' + FLAGS.set + '.txt')
        annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            try:
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                try:
                    xml = etree.fromstring(xml_str)
                except:
                    print('cannot for xml from string....')
            except:
                print('xml has some error.....')
                continue
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict)
            if tf_example != -1:
                writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()

