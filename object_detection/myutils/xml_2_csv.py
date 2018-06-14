import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf

tf.flags.DEFINE_string("data_dir",None, "Full root directory of annotations folder")
FLAGS = tf.app.flags.FLAGS

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            fname = root.find('filename').text
            filename = fname.split('.')[0]
            value = (filename,
                    int(root.find('size')[1].text),
                    int(root.find('size')[0].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text))
            xml_list.append(value)
    column_name = ['ImageName', 'height', 'width', 'category', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():

    assert FLAGS.data_dir, 'Root directory of Annotations folder does not exist'

    image_path = os.path.join(FLAGS.data_dir, 'Annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('annotations.csv', index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()