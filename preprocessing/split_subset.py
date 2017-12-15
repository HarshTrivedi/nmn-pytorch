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

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

annotation_file = os.path.join(
    root_dir, 'raw_data/Annotations/v2_mscoco_%s_annotations.json')
question_file = os.path.join(
    root_dir, 'raw_data/Questions/v2_OpenEnded_mscoco_%s_questions.json')

source_sets = []

selected_question_ids = set()
for source_set in source_sets:

    annotations = json.load(open(annotation_file % source_set))
    questions = json.load(open(question_file % source_set))

    question_type_to_question_ids = collections.defaultdict(list)
    for annotation in annotations["annotations"]:
        question_type_to_question_ids[annotation['question_type']].append(
            annotation['question_id'])

    for qtype in filter_question_types:
        selected_question_ids.add(question_type_to_question_ids[qtype])

random.shuffle(selected_question_ids)

total = len(selected_question_ids)
train_selected_ids = selected_question_ids[:((total * 2) / 4)]
test_selected_ids = selected_question_ids[((total * 2) / 4):(total * 3) / 4]
val_selected_ids = selected_question_ids[(total * 3) / 4:]

train_annotations, test_annotations, val_annotations = [], [], []
train_questions, test_questions, val_questions = [], [], []

for source_set in source_sets:

    for annotation in json.load(
            open(annotation_file % source_set))['annotations']:
        if annotation['question_id'] in train_selected_ids:
            train_annotations.append(annotation)
        elif annotation['question_id'] in test_selected_ids:
            test_annotations.append(annotation)
        elif annotation['question_id'] in val_selected_ids:
            val_annotations.append(annotation)

    for question in json.load(open(question_file % source_set))['questions']:
        if question['question_id'] in train_selected_ids:
            train_question.append(question)
        elif question['question_id'] in test_selected_ids:
            test_questions.append(question)
        elif question['question_id'] in val_selected_ids:
            val_questions.append(question)

with open(question_file % 'train-sub-split', 'w') as f:
    json.dump({'questions': train_questions}, f)
with open(question_file % 'test-sub-split', 'w') as f:
    json.dump({'questions': test_questions}, f)
with open(question_file % 'val-sub-split', 'w') as f:
    json.dump({'questions': val_questions}, f)

with open(annotation_file % 'train-sub-split', 'w') as f:
    json.dump({'annotations': train_annotations}, f)
with open(annotation_file % 'test-sub-split', 'w') as f:
    json.dump({'annotations': test_annotations}, f)
with open(annotation_file % 'val-sub-split', 'w') as f:
    json.dump({'annotations': val_annotations}, f)

train_image_ids = set()
test_image_ids = set()
val_image_ids = set()

for question in train_questions:
    train_image_ids.add(question['image_id'])

for question in test_questions:
    test_image_ids.add(question['image_id'])

for question in val_questions:
    val_image_ids.add(question['image_id'])

image_file = os.path.join(root_dir, 'raw_data/Images/%s/COCO_%s_%012d.jpg')
image_dir = os.path.join(root_dir, 'Images/%s/')

#### dump images
for image_id in selected_image_ids:
    # print image_id
    source_path = image_file % (image_set_source, image_set_source, image_id)
    target_path = image_file % (image_set_target, image_set_target, image_id)
    copyfile(source_path, target_path)
print "dumped images"
