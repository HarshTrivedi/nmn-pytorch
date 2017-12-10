# mostly taken from: https://gist.github.com/jacobandreas/897987ac03f8d4b9ea4b9e44affa00e7

import caffe
from collections import defaultdict
import numpy as np
import os
import sys
import matplotlib.image

# caffe.set_device(0)
# caffe.set_mode_gpu()

image_set = "train2014-sub"
# image_set = "val2014-sub"

root_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)),  '..')

IMAGE_ROOT = os.path.join( root_dir, 
              'raw_data/Images/{}/'.format(image_set) )
IMAGE_CONV_DEST = os.path.join( root_dir, 
              'image_features/preprocessed_data/{}/'.format(image_set) )
IMAGENET_MEAN_FILE = os.path.join(root_dir, 
              'preprocessing/ilsvrc_2012_mean.npy')
VGG_MODEL = os.path.join( root_dir, 
              'preprocessing/lib/VGG_ILSVRC_16_layers.caffemodel' )
VGG_DEF = os.path.join( root_dir, 
              'preprocessing/lib/VGG_ILSVRC_16_layers_deploy.prototxt' )
BATCH_SIZE = 32

net = caffe.Net(VGG_DEF,
                VGG_MODEL,
                caffe.TEST)

print net.blobs.keys()

all_image_names = os.listdir(IMAGE_ROOT)
all_image_names = [n for n in all_image_names if n[-3:] == "jpg"]

image_names_by_size = defaultdict(list)
for n in all_image_names:
  full_name = os.path.join(IMAGE_ROOT, n)
  try:
    image = caffe.io.load_image(full_name)
  except Exception as e:
    print >>sys.stderr, "unable to load image " + full_name
    continue
  width, height = image.shape[:2]
  image_names_by_size[width, height].append(n)

total_count = 0
for size, names in image_names_by_size.items():
  n_images = len(names)
  print ">", size, n_images
  i = 0
  while i < n_images:
    real_count = BATCH_SIZE if i + BATCH_SIZE < n_images else n_images - i
    names_here = [os.path.join(IMAGE_ROOT, names[i+j]) for j in range(real_count)]
    images = [caffe.io.load_image(name) for name in names_here]
    #images = [caffe.io.resize_image(image, (224,224)) for image in images]
    images = [caffe.io.resize_image(image, (448,448)) for image in images]

    net.blobs["data"].reshape(real_count, 3, images[0].shape[0], images[0].shape[1])

    transformer = caffe.io.Transformer({'data': net.blobs["data"].data.shape})
    transformer.set_transpose("data", (2, 0, 1))
    transformer.set_mean("data", np.load( IMAGENET_MEAN_FILE ).mean(1).mean(1))
    transformer.set_raw_scale("data", 255)
    transformer.set_channel_swap("data", (2,1,0))

    proc_images = [transformer.preprocess("data", image) for image in images]

    fw = net.forward(data=np.asarray(proc_images))
    embeddings = net.blobs["pool5"].data.copy()

    print ">>", embeddings.shape

    for j in range(real_count):
      print IMAGE_CONV_DEST + "/" + names[i+j]
      np.savez(IMAGE_CONV_DEST + "/" + names[i+j], embeddings[j,:,:,:])

    print total_count, i
    print

    total_count += real_count
    i += BATCH_SIZE

