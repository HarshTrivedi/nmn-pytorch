from lib.data_loader import *
import train_cmp_nn_vqa
import train_baseline_lstm
import train_baseline_lstm_img
import torch

### setup ###
model_name = 'cmp_nn_vqa'
train_set = 'train2014-sub'
test_set = 'test2014-sub'
num_questions = 10
print torch.cuda.is_available()

root_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname('.')
save_models_dir = os.path.join(root_dir, 'saved_models')
save_model_file = os.path.join(save_models_dir, model_name + '.pt')

results_dir = os.path.join(root_dir, 'results')

saved_mean_stds_file = os.path.join(
    root_dir,
    'preprocessed_data/image_features/mean-stds-{}.pickle'.format(train_set))
train_dataset, train_mean_stds = get_dataset(train_set, None,
                                             saved_mean_stds_file)
test_dataset, _ = get_dataset(test_set, train_mean_stds)

test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=4)

gpu_available = torch.cuda.is_available()
print "GPU Availibility: {}".format(
    gpu_available)  # GPU is necessary (for now!)
vqa_model = torch.load(save_model_file)

## show visualization of some num_questions ###
if model_name == 'cmp_nn_vqa':
    train_cmp_nn_vqa.visualize_model(vqa_model, test_dataloader, test_dataset,
                                     num_questions, False)

elif model_name == 'baseline_lstm_img':
    baseline_lstm_img.visualize_model(vqa_model, test_dataloader, test_dataset,
                                      num_questions, False)

elif model_name == 'baseline_lstm':
    baseline_lstm.visualize_model(vqa_model, test_dataloader, test_dataset,
                                  num_questions, False)
