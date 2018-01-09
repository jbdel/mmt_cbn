import tensorflow as tf
import numpy as np
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _string_feature(value):
    value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

dir_txt = "/home/jb/work/data/WMT17/2016/processed"
dir_img = "/home/jb/work/data/WMT17/flickr30k_images"
out_dir = "dataset_resnet"
num_per_file = 2300

for mode in ["train","val","test2016"]:
    source = []
    target = []
    img = []
    with open(dir_txt+'/'+mode+'.norm.tok.lc.bpe10000.en','r') as f:
      lines = f.readlines()
      for l in lines:
        source.append(l.strip())

    with open(dir_txt+'/'+mode+'.norm.tok.lc.bpe10000.de','r') as f:
      lines = f.readlines()
      for l in lines:
        target.append(l.strip())

    with open(dir_txt+'/'+mode+'_images.txt','r') as f:
      lines = f.readlines()
      for l in lines:
        img.append(l.strip())

    # if mode == "test2016":
    #     mode = "test"

    assert len(source) == len(target) == len(img),"nope"
    for i in range(len(source)):
      if i%num_per_file == 0:
        tfrecords_filename = dir_txt + "/"+ out_dir + '/'+ 'dataset_' + mode + str(int(i/num_per_file))+'.tfrecords'
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

      with tf.gfile.FastGFile(dir_img+"/"+img[i], "rb") as f:
          encoded_image = f.read()
          
      example = tf.train.Example(features=tf.train.Features(feature={
        'source': _string_feature(source[i]),
        'target': _string_feature(target[i]),
        'image': _bytes_feature(encoded_image),
      }))

      # print(example)
      # import sys
      # sys.exit()
      writer.write(example.SerializeToString())

    writer.close()


