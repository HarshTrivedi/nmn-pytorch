import train_cmp_nn_vqa
import train_baseline_lstm
import train_baseline_lstm_img
from lib.data_loader import * # ../ directory
import torch

root_dir = os.path.dirname(os.path.realpath(__file__))

train_set = "Train2014-sub" # std-means will be needed to pre-normalize demo set
demo_set = "Demo" # might want to shift demo to root_dir if load model doesn't work

question_file = os.path.join( root_dir, 
            'raw_data/Questions/v2_OpenEnded_mscoco_%s_questions.json' )

image_file = os.path.join( root_dir, 
            'raw_data/Images/%s/COCO_%s_%012d.jpg' )
image_dir  = os.path.join(  root_dir, 
            'Images/%s/' )

# annotation_file = os.path.join( root_dir, 
#             'raw_data/Annotations/v2_mscoco_%s_annotations.json' )
# Add extra argument in train_* files so that it doesn't break
# if annotations are not given. We don't really have any annotation!
# We don't want evaluation in case of demo

# Take data from here and place it in appropriate demo-set
# just have a copy of extract_image_vgg_features.py in lib/
# that can take input directory as argument. also expose a usable method from it
# use method from  lib.extract_image_vgg_features to process image
# and put output in preprecessed_data/image_features/ <demo_set> / ...

model_name = 'cmp_nn_vqa'

saved_model_path = os.path.join(root_dir, 'saved_model/{}.pt' .format(model_name) )
# add if __name__ == "__main__": in train_* files so that they can be loaded


# clear the demo directories: etg
# pick images and questions from input

saved_mean_stds_file = os.path.join(root_dir, 
                    'preprocessed_data/image_features/mean-stds-{}.pickle'.format(train_set) )
with open(saved_mean_stds_file, 'w') as f:
	train_mean_stds = pickle.load(saved_mean_stds_file)

demo_dataset, _ = get_dataset(demo_set, train_mean_stds)

model = torch.load(saved_model_path)

demo_dataloader   = DataLoader(demo_dataset,  batch_size=1, shuffle=True, num_workers=4)

if model_name == 'cmp_nn_vqa':
	result_dict  , _ = train_cmp_nn_vqa.test_model(model, demo_dataloader, demo_dataset, eval = False)
if model_name == 'baseline_lstm':
	result_dict  , _ = train_baseline_lstm.test_model(model, demo_dataloader, demo_dataset, eval = False)
if model_name == 'baseline_lstm_img':
	result_dict  , _ = train_baseline_lstm_img.test_model(model, demo_dataloader, demo_dataset, eval = False)


# use result_dict to show result on stdout
# store it in output/




