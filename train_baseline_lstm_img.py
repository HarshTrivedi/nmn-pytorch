from baseline_lstm_img_model import *
import torchvision.transforms as transforms
from lib.data_loader import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
import json
import os
import time
import logging




def train_model(model, datasetloader_dict, dataset_dict, loss_function, optimizer, num_epochs=50):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for batch_index, batch_datums in enumerate(datasetloader_dict[phase]):
                optimizer.zero_grad()

                ### forward (compute loss) ###
                # change this block Model Wise
                model.lstmModel.hidden = model.lstmModel.init_hidden() # you must do this!
                if use_gpu:
                    image_feat = Variable(batch_datums['image_feature_tensor'].cuda(), requires_grad = False)
                    actual_label_tensor = Variable(batch_datums['answer_index'].cuda(), requires_grad = False).view(-1)
                    token_sequence_tensor = Variable(batch_datums['question_token_ids'].cuda(), requires_grad = False)
                    answer_vector_tensor  = Variable(batch_datums['answer_vector_tensor'].cuda(), requires_grad = False)
                else:
                    image_feat = Variable(batch_datums['image_feature_tensor'], requires_grad = False)
                    actual_label_tensor = Variable(batch_datums['answer_index'], requires_grad = False).view(-1)
                    token_sequence_tensor = Variable(batch_datums['question_token_ids'], requires_grad = False)            
                    answer_vector_tensor  = Variable(batch_datums['answer_vector_tensor'], requires_grad = False)
                prediction_scores_tensor = model.forward(image_feat, token_sequence_tensor )

                loss = loss_function(prediction_scores_tensor, actual_label_tensor)
                loss_value = loss.data[0]


                ### Statistics ###
                topK = 3
                _, predictions = torch.max(prediction_scores_tensor.data, 1)
                prediction = predictions.view(-1)
                answer_vector = answer_vector_tensor.data.view(-1)
                top_predicted_indices = torch.topk( prediction_scores_tensor.data, topK, 1 )[1][0].tolist()
                correctness = max([ int(answer_vector[prediction]) for prediction in top_predicted_indices])
                running_corrects += correctness
                running_loss     += loss_value

                ### backward + optimize (if in training phase) ###
                if phase == 'train' and epoch > 0:
                    loss.backward()
                    optimizer.step()

                ### some debug logs ###
                if total % 1000 == 0:
                    logging.info('{} datums processed'.format(int(total)))

                total += len(batch_datums[batch_datums.keys()[0]])

            epoch_loss = running_loss / float(total)
            epoch_acc = running_corrects / float(total)

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, datasetloader, dataset):
    since = time.time()
    running_corrects = 0
    total = 0
    result_dict = {}
    for batch_index, batch_datums in enumerate(datasetloader):
        optimizer.zero_grad()

        ### forward (compute predictions) ###
        # change this block Model Wise
        model.lstmModel.hidden = model.lstmModel.init_hidden() # you must do this!
        if use_gpu:
            image_feat = Variable(batch_datums['image_feature_tensor'].cuda(), requires_grad = False)
            actual_label_tensor = Variable(batch_datums['answer_index'].cuda(), requires_grad = False).view(-1)
            token_sequence_tensor = Variable(batch_datums['question_token_ids'].cuda(), requires_grad = False)
            answer_vector_tensor  = Variable(batch_datums['answer_vector_tensor'].cuda(), requires_grad = False)
            question_id = Variable(batch_datums['question_id'].cuda(), requires_grad = False)
        else:
            image_feat = Variable(batch_datums['image_feature_tensor'], requires_grad = False)
            actual_label_tensor = Variable(batch_datums['answer_index'], requires_grad = False).view(-1)
            token_sequence_tensor = Variable(batch_datums['question_token_ids'], requires_grad = False)            
            answer_vector_tensor  = Variable(batch_datums['answer_vector_tensor'], requires_grad = False)
            question_id = Variable(batch_datums['question_id'], requires_grad = False)
        prediction_scores_tensor = model.forward(image_feat, token_sequence_tensor )


        ### Statistics ###
        topK = 3
        _, predictions = torch.max(prediction_scores_tensor.data, 1)
        prediction = predictions.view(-1)
        answer_vector = answer_vector_tensor.data.view(-1)
        top_predicted_indices = torch.topk( prediction_scores_tensor.data, topK, 1 )[1][0].tolist()
        correctness = max([ int(answer_vector[prediction]) for prediction in top_predicted_indices])
        running_corrects += correctness
        answers = [ dataset.answer_dict.idx2word(idx) for idx in top_predicted_indices]
        qid = int(question_id[0].data[0])
        dict_entry = {'question_id': qid, 'answers': answers, 'correctness': correctness}
        result_dict[qid] = dict_entry
        running_corrects += answer_vector[prediction]
        total += len(batch_datums[batch_datums.keys()[0]])


    time_elapsed = time.time() - since
    accuracy = running_corrects/float(total)
    logging.info('Testing complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    logging.info('Acc: {:4f}'.format(accuracy))
    return result_dict, accuracy


def visualize_model(model, datasetloader, dataset, num_questions=10):
    import matplotlib.pyplot as plt
    images_so_far = 0
    for batch_index, batch_datums in enumerate(datasetloader):

        ### forward (compute loss) ###
        # change this block Model Wise
        model.lstmModel.hidden = model.lstmModel.init_hidden() # you must do this!
        if use_gpu:
            image_feat = Variable(batch_datums['image_feature_tensor'].cuda(), requires_grad = False)
            actual_label_tensor = Variable(batch_datums['answer_index'].cuda(), requires_grad = False).view(-1)
            token_sequence_tensor = Variable(batch_datums['question_token_ids'].cuda(), requires_grad = False)
            answer_vector_tensor  = Variable(batch_datums['answer_vector_tensor'].cuda(), requires_grad = False)
        else:
            image_feat = Variable(batch_datums['image_feature_tensor'], requires_grad = False)
            actual_label_tensor = Variable(batch_datums['answer_index'], requires_grad = False).view(-1)
            token_sequence_tensor = Variable(batch_datums['question_token_ids'], requires_grad = False)            
            answer_vector_tensor  = Variable(batch_datums['answer_vector_tensor'], requires_grad = False)
        prediction_scores_tensor = model.forward(image_feat, token_sequence_tensor )

        #### show
        label_texts = []
        answer_index  = batch_datums['answer_index'][0].numpy()
        qtoken_ids    = batch_datums['question_token_ids'][0].numpy()
        answer_vector = batch_datums['answer_vector_tensor'][0].numpy().reshape([-1])
        qid           = int(batch_datums['question_id'][0].numpy())

        question = [ q for q in dataset.questions_dict if q['question_id'] == qid ][0]
        question_text = question['question']
        title_text = "Q: {}".format( question_text )

        layout     = dataset.qid2layout_dict[ str(question['question_id']) ]
        label_texts.append("L: {}".format( layout ))

        topK = 3
        top_predicted_indices = torch.topk( prediction_scores_tensor.data, topK, 1 )[1][0].tolist()
        predicted_answers = [ dataset.answer_dict.idx2word(answer_id) for answer_id in top_predicted_indices]
        label_texts.append( 'A: ' + ' ; '.join(predicted_answers) )

        image_id = question['image_id']
        raw_image_file = os.path.join(root_dir, 
                         'raw_data/Images/%s/COCO_%s_%012d.jpg' % (dataset.set_name, dataset.set_name, int(image_id) )  )

        plt.figure()
        image_data = plt.imread( image_path, format='jpg' )
        plt.title(question_text)
        plt.xlabel('\n'.join(label_texts))
        plt.imshow(image_data)
        plt.show()

        images_so_far += 1
        if images_so_far > num_questions:
            break

###################################################################################


if __name__ == "__main__":

    ######## Set up (Starts) #########

    model_name   = 'baseline_lstm_img'
    train_set = 'train2014-sub'
    val_set  = 'val2014-sub'

    root_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(root_dir, 'logs')
    logging.basicConfig(    filename= os.path.join(log_dir, model_name + '.log'), 
                            level=logging.INFO, 
                            format="%(asctime)s:%(message)s", 
                            filemode='w' )

    results_dir = os.path.join( root_dir, 'results' )
    train_results_file = os.path.join( results_dir, '_'.join([ model_name, train_set, train_set ]) + '.json' )
    val_results_file  = os.path.join( results_dir, '_'.join([ model_name, train_set, val_set ]) + '.json' )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    saved_mean_stds_file = os.path.join(root_dir, 
                        'preprocessed_data/image_features/mean-stds-{}.pickle'.format(train_set) )

    annotation_file = os.path.join( root_dir, 
                'raw_data/Annotations/v2_mscoco_%s_annotations.json' )

    vocab_answer_file = os.path.join( root_dir,
                'preprocessed_data/vqa_answers.txt')


    save_models_dir = os.path.join(root_dir, 'saved_models')
    save_model_file = os.path.join(save_models_dir, model_name + '.pt')
    if not os.path.exists(save_models_dir):
        os.makedirs(save_models_dir)


    use_gpu = torch.cuda.is_available()
    logging.info("GPU availibility: {}".format(use_gpu))

    if use_gpu:
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)    

    ######## Set up (Ends) #########


    train_dataset, train_mean_stds  = get_dataset(train_set, None, saved_mean_stds_file )
    val_dataset, _                  = get_dataset(val_set, train_mean_stds)
    logging.info("Datasets Loaded")

    dataset = { 'train': train_dataset, 'val'  : val_dataset }

    train_annotation_dict = json.load(open(annotation_file % train_set))
    val_annotation_dict   = json.load(open(annotation_file % val_set))

    D_img, D_txt, D_map, D_hidden = 512, 300, 1024, 1024
    with open(vocab_answer_file, 'r') as f:
        D_ans_choices = len( [e for e in f.readlines() if e.strip() != ""] )

    question_word_embeddings_np = dataset['train'].question_embedding_matrix
    question_vocab_size         = dataset['train'].question_vocab_dict.num_vocab

    ######
    # change this block Model Wise
    model = VGABaseModel(   D_img, 
                            D_txt, 
                            D_map, 
                            D_hidden, 
                            question_word_embeddings_np, 
                            question_vocab_size,  
                            D_ans_choices)
    ######

    if use_gpu:
        model.cuda()

    loss_function = torch.nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(parameters)
    logging.info('Built Model')

    train_dataloader = DataLoader(dataset['train'], batch_size=1, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(dataset['val'] ,  batch_size=1, shuffle=True, num_workers=4)
    dataloader = { 'train': train_dataloader, 'val' : val_dataloader }
    logging.info('Dataloader Loaded')

    trained_vqa_model = train_model(model, dataloader, dataset, loss_function, optimizer, num_epochs=50)
    logging.info('Training Complete')

    torch.save(trained_vqa_model, save_model_file)
    logging.info('Saved Model')


    train_result_dict, train_accuracy = test_model(trained_vqa_model, train_dataloader, train_dataset)
    val_result_dict  , val_accuracy   = test_model(trained_vqa_model, val_dataloader , val_dataset)

    with open(train_results_file, 'w') as f:
        json.dump(train_result_dict, f)

    with open(val_results_file, 'w') as f:
        json.dump(val_result_dict, f)

    logging.info('saved result jsons')

    # MatplotLib would pop to show
    visualize_model(trained_vqa_model, val_dataloader, val_dataset, num_questions=10)




