import json
from pprint import pprint
from shutil import copyfile
import copy
import collections
import os

# setting (start)#
image_set_source = "val2014"
image_set_target = "val2014-sub"
filter_question_types = ["where is the", "where are the"]
# setting (end)#

root_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)),  '..')

annotation_file = os.path.join( root_dir, 
            'raw_data/Annotations/v2_mscoco_%s_annotations.json' )
question_file = os.path.join( root_dir, 
            'raw_data/Questions/v2_OpenEnded_mscoco_%s_questions.json' )
image_file = os.path.join( root_dir, 
            'raw_data/Images/%s/COCO_%s_%012d.jpg' )
image_dir  = os.path.join(  root_dir, 
            'Images/%s/' )

if not os.path.exists(image_dir % image_set_target):
    os.makedirs(image_dir)

annotations = json.load(open( annotation_file % image_set_source ))
questions   = json.load(open( question_file % image_set_source ))

question_type_to_question_ids = collections.defaultdict(list)
for annotation in annotations["annotations"]:
    question_type_to_question_ids[annotation['question_type']].append(annotation['question_id'])

selected_question_ids = []
for qtype in filter_question_types:
    selected_question_ids.extend(question_type_to_question_ids[qtype])
print "{} selected question ids".format(len(selected_question_ids))

selected_image_ids = set()
for question in questions['questions']:
    if question['question_id'] in selected_question_ids:
        selected_image_ids.add(question['image_id'])
print "{} selected image ids".format(len(selected_image_ids))

#### subselect questions
for index, annotation in enumerate(annotations['annotations']): 
    if annotation['question_id'] not in selected_question_ids:
        annotations['annotations'][index] = None
annotations['annotations'] = filter(bool, annotations['annotations'])
print "built sub annotations"

#### subselect annotations
for index, question in enumerate(questions['questions']): 
    if question['question_id'] not in selected_question_ids:
        questions['questions'][index] = None
questions['questions'] = filter(bool, questions['questions'])
print "built sub questions"

#### dump questions
with open( question_file % image_set_target, 'w' ) as f:
    json.dump( questions, f )
print "dumped questions"

#### dump annotations
with open( annotation_file % image_set_target, 'w' ) as f:
    json.dump( annotations, f )
print "dumped annotations"

#### dump images
for image_id in selected_image_ids:
    # print image_id
    source_path = image_file % (image_set_source, image_set_source, image_id)
    target_path = image_file % (image_set_target, image_set_target, image_id)
    copyfile(source_path, target_path)
print "dumped images"
