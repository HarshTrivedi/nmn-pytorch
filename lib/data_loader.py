from torch.utils.data import Dataset, DataLoader
import cPickle as pickle
import torchvision.transforms as transforms
import torch
import json
import lib.load_vocab
import numpy as np
import os
import ast


class VqaDataset(Dataset):

    def __init__(self,
                 set_name,
                 vocab_answer_file,
                 vocab_module_file,
                 vocab_question_file,
                 wordvec_file,
                 annotation_file,
                 question_file,
                 module_layout_file,
                 image_feature_file,
                 transform=None):

        with open(question_file) as f:
            questions_dict = json.load(f)['questions']

        question_vocab_dict = lib.load_vocab.VocabDict(vocab_question_file)
        module_vocab_dict = lib.load_vocab.VocabDict(vocab_module_file)
        answer_dict = lib.load_vocab.VocabDict(vocab_answer_file)

        valid_answer_set = set(answer_dict.word_list)
        answer_choices_count = len(valid_answer_set)

        with open(wordvec_file) as f:
            question_embedding_matrix = np.load(f)

        self.set_name = set_name
        self.questions_dict = questions_dict
        self.answer_dict = answer_dict

        self.module_vocab_dict = module_vocab_dict
        self.question_vocab_dict = question_vocab_dict

        self.question_embedding_matrix = question_embedding_matrix

        self.valid_answer_set = valid_answer_set
        self.answer_choices_count = answer_choices_count

        self.transform = transform

        self.module_layout_file = module_layout_file  # Not Needed Actually
        self.annotation_file = annotation_file  # Not Needed Actually

        qid2layout_dict = {}
        with open(module_layout_file) as f:
            for line in f.readlines():
                if line.strip() != "":
                    qid, layout_str = line.strip().split('\t')
                    layout = ast.literal_eval(layout_str)
                    qid2layout_dict[qid] = layout
        self.qid2layout_dict = qid2layout_dict

        with open(annotation_file) as f:
            annotations = json.load(f)["annotations"]
            qid2annotation_dict = {
                annotation['question_id']: annotation
                for annotation in annotations
            }
        self.qid2annotation_dict = qid2annotation_dict

        self.image_feature_file = image_feature_file
        ### Fix this

        self.image_feat_D = 512
        self.image_feat_H = 14
        self.image_feat_W = 14
        self.wordvec_size = 300

    def __len__(self):
        return len(self.questions_dict)

    def __getitem__(self, idx):

        question = self.questions_dict[idx]
        question_id = question['question_id']
        image_id = question['image_id']
        question_str = question['question']
        question_tokens = question_str.strip().lower().replace('?', '').split()

        layout = self.qid2layout_dict[str(question_id)]

        answer_vector = np.zeros([self.answer_choices_count, 1])

        annotation = self.qid2annotation_dict[question_id]
        valid_answers = [
            answer["answer"] for answer in annotation['answers']
            if answer["answer"] in self.valid_answer_set
        ]
        if len(valid_answers) == 0:
            valid_answers = ['<unk>']

        for valid_answer in valid_answers:
            answer_vector[self.answer_dict.word2idx(valid_answer), 0] = 1.0

        use_answer = np.random.choice(valid_answers)
        answer_index = self.answer_dict.word2idx(use_answer)

        image_feature_tensor = np.load(
            self.image_feature_file % image_id)['arr_0']
        question_token_ids = np.array([
            self.question_vocab_dict.word2idx(token)
            for token in question_tokens
        ])

        datum = {
            'image_feature_tensor': image_feature_tensor,
            'layout': layout,
            'answer_index': answer_index,
            'question_token_ids': question_token_ids,
            'answer_vector_tensor': answer_vector,
            'question_id': question_id
        }

        if self.transform:
            datum = self.transform(datum)

        return datum

    def show_batch(self, sample_batched):

        answer_index = sample_batched['answer_index'][0].numpy()
        qtoken_ids = sample_batched['question_token_ids'][0].numpy()
        answer_vector = sample_batched['answer_vector_tensor'][
            0].numpy().reshape([-1])
        qid = int(sample_batched['question_id'][0].numpy())

        question = [q for q in self.questions_dict
                    if q['question_id'] == qid][0]
        layout = self.qid2layout_dict[str(question['question_id'])]

        annotation = self.qid2annotation_dict[question['question_id']]
        answers_text = [a['answer'] for a in annotation['answers']]

        actual_answers_ids = [
            self.answer_dict.word2idx(a['answer'])
            for a in annotation['answers']
        ]
        constructed_answer_ids = list(np.nonzero(answer_vector))[0]
        constructed_answers_text = [
            self.answer_dict.idx2word(answer_id)
            for answer_id in constructed_answer_ids
        ]

        question_text = question['question']
        image_id = question['image_id']
        question_text_constructed = [
            self.question_vocab_dict.idx2word(token_id)
            for token_id in qtoken_ids
        ]

        print "---------------------"
        print 'question text actual:'
        print question_text
        print 'question text constructed'
        print question_text_constructed

        print 'image path:'
        print self.image_feature_file % image_id

        print 'answer ids actual:'
        print actual_answers_ids
        print 'answer ids constructed:'
        print constructed_answer_ids

        print 'answers text actual:'
        print answers_text
        print 'answers text constructed:'
        print constructed_answers_text

        print 'layout:'
        print layout
        print '---------------------'

    @staticmethod
    def get_image_feature_mean_and_stds(question_file, image_feature_file,
                                        saved_mean_stds_file):

        image_feat_D = 512
        image_feat_H = 14
        image_feat_W = 14

        if os.path.exists(saved_mean_stds_file):
            means, stdevs = pickle.load(open(saved_mean_stds_file, 'rb'))
            return [means, stdevs]
        else:
            print 'precomputing data mean/stds for normalization'
            with open(question_file) as f:
                questions_dict = json.load(f)['questions']

            image_ids = set()
            for question in questions_dict:
                image_ids.add(question['image_id'])

            images_count = len(image_ids)
            vals = np.zeros(
                [image_feat_D, images_count * image_feat_H * image_feat_W])

            filled_upto = 0
            for done, image_id in enumerate(image_ids):
                if done % 1000 == 0: print done
                image_feature_tensor = np.load(
                    open(image_feature_file % image_id))['arr_0']
                for channel in range(image_feat_D):
                    vector = image_feature_tensor[channel, :, :].ravel()  #14X14
                    vals[channel, filled_upto:(
                        filled_upto + (image_feat_H * image_feat_W))] = vector
                filled_upto += (image_feat_H * image_feat_W)

            means = []
            stdevs = []
            for i in range(image_feat_D):
                means.append(np.mean(vals[i, :]))
                stdevs.append(np.std(vals[i, :]))

            with open(saved_mean_stds_file, 'wb') as f:
                pickle.dump([means, stdevs], f)
            return means, stdevs


class ToTensor(object):

    def __call__(self, datum):
        image_feature_tensor = datum['image_feature_tensor']
        layout = datum['layout']
        answer_index = datum['answer_index']
        question_token_ids = datum['question_token_ids']
        answer_vector_tensor = datum['answer_vector_tensor']
        question_id = datum['question_id']

        torch_datum = {
            'image_feature_tensor': torch.from_numpy(image_feature_tensor),
            'layout': layout,
            'answer_index': torch.from_numpy(np.array([answer_index])),
            'question_token_ids': torch.from_numpy(question_token_ids),
            'answer_vector_tensor': torch.from_numpy(answer_vector_tensor),
            'question_id': torch.from_numpy(np.array([question_id]))
        }
        return torch_datum


class Normalize(object):

    def __init__(self, means, stdevs):
        self.means = means
        self.stdevs = stdevs

    def __call__(self, torch_datum):

        image_feature_tensor = torch_datum['image_feature_tensor']
        layout = torch_datum['layout']
        answer_index = torch_datum['answer_index']
        question_token_ids = torch_datum['question_token_ids']
        answer_vector_tensor = torch_datum['answer_vector_tensor']
        question_id = torch_datum['question_id']

        img_depth = image_feature_tensor.size()[0]
        tensor_normalize = transforms.Normalize(
            mean=self.means, std=self.stdevs)
        normalized_image_feature_tensor = tensor_normalize(image_feature_tensor)

        normalized_torch_datum = {
            'image_feature_tensor': normalized_image_feature_tensor,
            'layout': str(layout),
            'answer_index': answer_index,
            'question_token_ids': question_token_ids,
            'answer_vector_tensor': answer_vector_tensor,
            'question_id': question_id
        }
        return normalized_torch_datum


def get_dataset(image_set, use_mean_stds, saved_mean_stds_file=None):

    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

    vocab_answer_file = os.path.join(root_dir,
                                     'preprocessed_data/vqa_answers.txt')
    vocab_module_file = os.path.join(root_dir,
                                     'preprocessed_data/vocabulary_modules.txt')
    vocab_question_file = os.path.join(root_dir,
                                       'preprocessed_data/vocabulary_vqa.txt')
    wordvec_file = os.path.join(root_dir,
                                'preprocessed_data/vocabulary_vqa_glove.npy')

    annotation_file = os.path.join(
        root_dir, 'raw_data/Annotations/v2_mscoco_%s_annotations.json')
    question_file = os.path.join(
        root_dir, 'raw_data/Questions/v2_OpenEnded_mscoco_%s_questions.json')
    module_layout_file = os.path.join(
        root_dir,
        'preprocessed_data/layouts/module_layouts/questions_module_layouts_%s.txt'
    )
    image_feature_file = os.path.join(
        root_dir, 'preprocessed_data/image_features/%s/COCO_%s_%%012d.jpg.npz')

    if use_mean_stds:  # In case of Test/Val (stats would be given)
        means, stdevs = use_mean_stds
    else:  # In case of Train (we will need to compute)
        means, stdevs = VqaDataset.get_image_feature_mean_and_stds(
            question_file % image_set,
            image_feature_file % (image_set, image_set), saved_mean_stds_file)
    vqa_dataset = VqaDataset(
        image_set,
        vocab_answer_file,
        vocab_module_file,
        vocab_question_file,
        wordvec_file,
        annotation_file % image_set,
        question_file % image_set,
        module_layout_file % image_set,
        image_feature_file % (image_set, image_set),
        transform=transforms.Compose([ToTensor(),
                                      Normalize(means, stdevs)]))

    return vqa_dataset, [means, stdevs]
