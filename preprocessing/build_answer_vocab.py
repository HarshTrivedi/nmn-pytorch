import json
import os

root_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)),  '..')

print annotation_file

#Only Training Should be used here!
image_set = "train2014-sub" 
annotation_file = os.path.join(root_dir, 
	'raw_data/Annotations/v2_mscoco_%s_annotations.json' % image_set  )


with open(annotation_file) as f:
    annotations = json.load(f)["annotations"]

with open( os.path.join( root_dir, 'preprocessed_data/vqa_answers_original.txt' ) ) as f:
    original_valid_answers = [ line.strip() for line in f.readlines() if line.strip()]

all_valid_answers = set(['<unk>'])
for annotation in annotations:    
    valid_answers = [ answer["answer"] for answer in annotation['answers'] if answer["answer_confidence"] == 'yes' ]    
    # all_valid_answers.add( annotation['multiple_choice_answer'] )
    for answer in valid_answers:
        if answer in original_valid_answers:
            all_valid_answers.add( answer )


with open( os.path.join( root_dir, 'preprocessed_data/vqa_answers_original.txt' ), 'w') as f:
    f.write( '\n'.join(all_valid_answers) )


