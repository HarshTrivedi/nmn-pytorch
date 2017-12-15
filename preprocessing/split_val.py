import json
from pprint import pprint
from shutil import copyfile
import copy
import collections
import os
import random

# setting (start)#
val_set = "val2014-sub"
test_set = "test2014-sub"
# setting (end)#

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

annotation_file = os.path.join(
    root_dir, 'raw_data/Annotations/v2_mscoco_%s_annotations.json')
question_file = os.path.join(
    root_dir, 'raw_data/Questions/v2_OpenEnded_mscoco_%s_questions.json')

questions = json.load(open(question_file % val_set))
annotations = json.load(open(annotation_file % val_set))

val_questions = {'questions': []}
test_questions = {'questions': []}

val_annotations = {'annotations': []}
test_annotations = {'annotations': []}

question_ids = [question['question_id'] for question in questions['questions']]
print "{} questions in validation".format(question_ids)

random.shuffle(question_ids)

total = len(question_ids)

val_qids = question_ids[:(total / 2)]
test_qids = question_ids[(total / 2):]
print "splitting that to: {} val, {} test".format(len(val_qids), len(test_qids))

val_questions['questions'] = [
    question for question in questions['questions']
    if question['question_id'] in val_qids
]
test_questions['questions'] = [
    question for question in questions['questions']
    if question['question_id'] in test_qids
]

val_annotations['annotations'] = [
    annotation for annotation in annotations['annotations']
    if annotation['question_id'] in val_qids
]
test_annotations['annotations'] = [
    annotation for annotation in annotations['annotations']
    if annotation['question_id'] in test_qids
]

test_image_ids = set()
for question in test_questions['questions']:
    test_image_ids.add(question['image_id'])

print "dumping questions"
with open(question_file % val_set, 'w') as f:
    json.dump(val_questions, f)
with open(question_file % test_set, 'w') as f:
    json.dump(test_questions, f)

print "dumping annotations"
with open(annotation_file % val_set, 'w') as f:
    json.dump(val_annotations, f)
with open(annotation_file % test_set, 'w') as f:
    json.dump(test_annotations, f)

# Uncomment This Later
# print "dumping images"
# No NEED TO TRANSFER IMAGES FOR NOW
# image_file = os.path.join( root_dir,
#             'raw_data/Images/%s/COCO_%s_%012d.jpg' )
# image_dir  = os.path.join(  root_dir,
#             'Images/%s/' )

# # make dir if doesn't exist ...
# if not os.path.exist(image_dir):
#     os.makedirs(image_dir)

# #### move images
# for image_id in test_image_ids:
#     source_path = image_file % (val_set,  val_set,  image_id)
#     target_path = image_file % (test_set, test_set, image_id)
#     movefile(source_path, target_path)
# print "moved images"

# Comment This Later
# NEED TO TRANSFER FEATURES:
# print "dumping features"
image_feature_file = os.path.join(
    root_dir, 'preprocessed_data/image_features/%s/COCO_%s_%012d.jpg.npz')
image_feature_dir = os.path.join(
    root_dir, 'preprocessed_data/image_features/%s/' % test_set)

# make dir if doesn't exist ...
if not os.path.exists(image_feature_dir):
    os.makedirs(image_feature_dir)

#### move images
for image_id in test_image_ids:
    source_path = image_feature_file % (val_set, val_set, int(image_id))
    target_path = image_feature_file % (test_set, test_set, int(image_id))
    copyfile(source_path, target_path)
