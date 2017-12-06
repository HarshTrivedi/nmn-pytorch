import torch
import torch.nn.functional as F
import numpy as np

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)    
    

class LSTMModel(torch.nn.Module):

    def __init__(self, question_word_embeddings_np, embedding_dim, hidden_dim, vocab_size, D_ans_choices ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.question_word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.question_word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(question_word_embeddings_np))
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


class VGABaseModel(torch.nn.Module):

    def __init__(self, D_img, D_txt, D_map, D_hidden, question_word_embeddings_np, question_vocab_size, ans_choices_size):
        super(VGABaseModel, self ).__init__()
        self.squeeze_wordvec_to_map_d = torch.nn.Linear(D_hidden, D_map)
        self.squeeze_image_to_map_d   = torch.nn.Linear(D_img, D_map)
        self.map_to_answer_choices    = torch.nn.Linear(D_map, D_ans_choices)    
        self.lstmModel = LSTMModel(question_word_embeddings_np, D_txt, D_hidden, question_vocab_size, ans_choices_size )
        
    def forward(self, image_feat, token_sequence_tensor ):

        image_feat_mapped = self.squeeze_image_to_map_d(image_feat.sum(2).sum(2))
        lstm_encoded = self.lstmModel(token_sequence_tensor)

        lstm_encoded_mapped = self.squeeze_wordvec_to_map_d(lstm_encoded)
        elementwise_multiplication = torch.mul( lstm_encoded_mapped, image_feat_mapped )

        normalized_elementwise_multiplication = F.normalize(  elementwise_multiplication , p=2, dim=1)
        scores = self.map_to_answer_choices(normalized_elementwise_multiplication)

        return scores

