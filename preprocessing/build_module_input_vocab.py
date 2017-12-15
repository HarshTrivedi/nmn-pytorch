import ast
import os

image_set = "train2014-sub"  #Only Training Should be used here!

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def flatten(lol):
    return sum([flatten(l)
                for l in lol], []) if isinstance(lol, (tuple, list)) else [lol]


module_layouts_dir = os.path.join(root_dir,
                                  'preprocessed_data/layouts/module_layouts')
module_layout_file = os.path.join(
    module_layouts_dir, 'questions_module_layouts_{}.txt'.format(image_set))

module_vocab_file = os.path.join(root_dir,
                                 'preprocessed_data/vocabulary_modules.txt')

module_instance_vocab = set(['<unk>'])
with open(module_layout_file) as f:
    for line in f.readlines():
        if line.strip() != "":
            qid, layouts_str = line.strip().split('\t')
            layouts = ast.literal_eval(layouts_str)
            for token in set(flatten(layouts)) - set(
                ["Find", "Describe", "Transform", "And", "Or"]):
                module_instance_vocab.add(token)

with open(module_vocab_file, 'w') as f:
    f.write('\n'.join(module_instance_vocab))
