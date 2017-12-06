import cPickle as pickle
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np

use_gpu = torch.cuda.is_available()

if use_gpu:
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)

class FindModule( torch.nn.Module ):

    def __init__(self, D_img, D_txt, D_map):
        super(FindModule, self).__init__()

        self.squeeze_wordvec_to_map_d = torch.nn.Linear(D_txt, D_map)    
        self.squeeze_image_to_map_d   = torch.nn.Conv2d(D_img, D_map, 1, stride=1, padding=0, bias=True)
        self.squeeze_to_attention   = torch.nn.Conv2d(D_map, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, wordvec, image_feat ):
        
        wordvec_mapped = self.squeeze_wordvec_to_map_d( wordvec )
        wordvec_mapped = wordvec_mapped.view((1,-1,1,1))
        image_feat_mapped = self.squeeze_image_to_map_d( image_feat )      
        elementwise_multiplication = torch.mul(image_feat_mapped, wordvec_mapped)
        normalized_elementwise_multiplication = F.normalize(  elementwise_multiplication , p=2, dim=1)
        attention_grid = self.squeeze_to_attention(normalized_elementwise_multiplication)

        return attention_grid


class AndModule( torch.nn.Module ):

    def __init__(self):
        super(AndModule, self).__init__()

    def forward(self, attention_grid1, attention_grid2 ):
        return torch.min( attention_grid1, attention_grid2 )

class OrModule( torch.nn.Module ):

    def __init__(self):
        super(OrModule, self).__init__()

    def forward(self, attention_grid1, attention_grid2 ):
        return torch.max( attention_grid1, attention_grid2 )


class DescribeModule( torch.nn.Module ):

    def __init__(self, D_txt, D_img, D_map, D_hidden, question_vocab_size, D_ans_choices, question_word_embeddings_np):
        super(DescribeModule, self).__init__()

        self.squeeze_wordvec_to_map_d = torch.nn.Linear(D_txt, D_map)
        self.squeeze_attended_image_to_map_d = torch.nn.Linear(D_img, D_map)
        self.map_to_answer_choices = torch.nn.Linear(D_map, D_ans_choices)    

        # How to use this in Describe Module!??
        # LSTM here
        self.LSTMEncoder = LSTMEncoder(question_word_embeddings_np, D_txt, D_hidden, question_vocab_size ) 

    def forward(self, attention_grid, module_wordvec, image_feat, token_sequence_tensor ):

        wordvec_mapped = self.squeeze_wordvec_to_map_d( module_wordvec )
        wordvec_mapped = wordvec_mapped.view((1,-1))

        softmax_2d = torch.nn.Softmax2d()
        normalized_attention_grid = softmax_2d( attention_grid )

        attended_image_feat = torch.mul(normalized_attention_grid, image_feat)

        image_prob_vector = attended_image_feat.sum(2).sum(2) # sum over height and width. Retain depth (to sum attention probabilities).
        attention_feat_mapped = self.squeeze_attended_image_to_map_d(image_prob_vector)
        

        elementwise_multiplication = torch.mul( attention_feat_mapped, wordvec_mapped )

        # LSTM here
        last_hidden_state = self.LSTMEncoder( token_sequence_tensor ) 
        # LSTM here
        elementwise_multiplication = torch.mul( elementwise_multiplication, last_hidden_state )

        normalized_elementwise_multiplication = F.normalize(  elementwise_multiplication , p=2, dim=1)
        scores = self.map_to_answer_choices(normalized_elementwise_multiplication)
        
        return scores
        # softmax = torch.nn.Softmax()
        # return softmax(scores)


class TransformModule( torch.nn.Module ):

    def __init__(self, D_txt, D_img, D_map ):
        super(TransformModule, self).__init__()

        self.squeeze_wordvec_to_map_d        = torch.nn.Linear(D_txt, D_map)
        self.squeeze_attended_image_to_map_d = torch.nn.Linear(D_img, D_map)
        self.squeeze_image_to_map_d          = torch.nn.Conv2d(D_img, D_map, 1, stride=1, padding=0, bias=True)
        self.squeeze_to_attention            = torch.nn.Conv2d(D_map, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, input_attention_grid, wordvec, image_feat ):

        wordvec_mapped = self.squeeze_wordvec_to_map_d( wordvec )
        wordvec_mapped = wordvec_mapped.view((1,-1,1,1))

        image_feat_mapped = self.squeeze_image_to_map_d( image_feat )

        softmax_2d = torch.nn.Softmax2d()
        normalized_attention_grid = softmax_2d( input_attention_grid )
        attended_image_feat = torch.mul(normalized_attention_grid, image_feat)
        image_prob_vector = attended_image_feat.sum(2).sum(2) # sum over height and width. Retain depth (to sum attention probabilities).
        attention_feat_mapped = self.squeeze_attended_image_to_map_d(image_prob_vector)
        attention_feat_mapped = attention_feat_mapped.view((1,-1,1,1))
        
        elementwise_multiplication = torch.mul( attention_feat_mapped, wordvec_mapped )
        elementwise_multiplication = torch.mul( elementwise_multiplication, image_feat_mapped )
        normalized_elementwise_multiplication = F.normalize(  elementwise_multiplication , p=2, dim=1)
        
        output_attention_grid = self.squeeze_to_attention(normalized_elementwise_multiplication)

        return output_attention_grid


class LSTMEncoder(torch.nn.Module):

    def __init__(self, question_work_embeddings_np, embedding_dim, hidden_dim, vocab_size ):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.question_word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.question_word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(question_work_embeddings_np))
        self.question_word_embeddings.weight.requires_grad=False
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):        
        if use_gpu:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input_token_ids_sequence):
        embeds = self.question_word_embeddings(input_token_ids_sequence)
        lstm_out, self.hidden = self.lstm( embeds.view(input_token_ids_sequence.size()[1], 1, -1), self.hidden)
        return self.hidden[0].view(1, -1)


class DynamicModularNet(torch.nn.Module):
    def __init__(self, D_img, D_txt, D_map, D_hidden, question_word_embeddings_np, question_vocab_size, module_vocab_size, ans_choices_size):
        super(DynamicModularNet, self ).__init__()

        self.module_word_embeddings = torch.nn.Embedding(module_vocab_size, D_txt)

        self.Find       = FindModule(D_img, D_txt, D_map)
        self.Describe   = DescribeModule(D_txt, D_img, D_map, D_hidden, question_vocab_size, ans_choices_size, question_word_embeddings_np )
        self.Transform  = TransformModule(D_txt, D_img, D_map)
        self.And        = AndModule()
        # self.Or         = OrModule()
        
    def forward(self, Layout, image_feat, token_sequence_tensor, module_vocab_dict ):

        # Have idea from here: https://github.com/dasguptar/treelstm.pytorch/blob/master/model.py
        
        if Layout[0] == 'Find':
            # Find
            root_type, root_label = 'Find', Layout[1]
            if root_label not in module_vocab_dict.word_list:    root_label = '<unk>'
            root_label_id = module_vocab_dict.word2idx(root_label)
            # label_vec = torch.index_select(embedding_matrix, 0, torch.LongTensor([root_label_id]) )
            if use_gpu:
                label_vec = self.module_word_embeddings( Variable(torch.LongTensor([root_label_id]).cuda(), requires_grad = False) ).view(-1)
            else:
                label_vec = self.module_word_embeddings( Variable(torch.LongTensor([root_label_id]), requires_grad = False) ).view(-1)
            return self.Find(label_vec, image_feat )

        else:

            # print Layout
            LayoutRoot = Layout[0]
            LayoutSubtree = Layout[1]
            root_type, root_label = LayoutRoot

            if root_type == "Describe":
                if root_label not in module_vocab_dict.word_list:    root_label = '<unk>'
                root_label_id = module_vocab_dict.word2idx(root_label)

                if use_gpu:
                    label_vec = self.module_word_embeddings( Variable(torch.LongTensor([root_label_id]).cuda(), requires_grad = False )).view(-1)
                else:
                    label_vec = self.module_word_embeddings( Variable(torch.LongTensor([root_label_id]), requires_grad = False )).view(-1)

                subtree_attention = self.forward(LayoutSubtree, image_feat, token_sequence_tensor, module_vocab_dict)
                return self.Describe( subtree_attention, label_vec, image_feat, token_sequence_tensor )

            if root_type == "Transform":
                if root_label not in module_vocab_dict.word_list:    root_label = '<unk>'
                root_label_id = module_vocab_dict.word2idx(root_label)

                if use_gpu:
                    label_vec = self.module_word_embeddings( Variable(torch.LongTensor([root_label_id]).cuda(), requires_grad = False) ).view(-1)
                else:
                    label_vec = self.module_word_embeddings( Variable(torch.LongTensor([root_label_id]), requires_grad = False) ).view(-1)
                subtree_attention = self.forward(LayoutSubtree, image_feat, token_sequence_tensor, module_vocab_dict)
                return self.Transform( subtree_attention, label_vec, image_feat )

            if root_type == "And":
                child_attentions = [ self.forward(sub_node, image_feat, module_vocab_dict) for sub_node in LayoutSubtree ]
                return self.And( child_attentions[0], child_attentions[1]  )




