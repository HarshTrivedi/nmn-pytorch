import sys
from lib.vqa import VQA
from lib.vqaEval import VQAEval
import matplotlib.pyplot as plt
import os
import json

model_name = 'cmp_nn_vqa'
root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

train_set_name = 'train2014-sub'
test_set_name = 'test2014-sub'

annotation_file = os.path.join(
    root_dir,
    'raw_data/Annotations/v2_mscoco_%s_annotations.json' % test_set_name)

question_file = os.path.join(
    root_dir,
    'raw_data/Questions/v2_OpenEnded_mscoco_%s_questions.json' % test_set_name)

resule_file = os.path.join(root_dir, 'results/%s_%s_%s.json' %
                           (model_name, train_set_name, test_set_name))

vqa = VQA(annotation_file, question_file)
resule_file = vqa.loadRes(resFile, question_file)
vqaEval = VQAEval(vqa, resule_file)
vqaEval.evaluate()

print "---------------------------------"
print "Overall Top 1 Accuracy is: %.02f" % (vqaEval.accuracy['overall'])
print "Overall Top 3 Accuracy is: %.02f" % (vqaEval.accuracy['overall-top-3'])
print "---------------------------------"
