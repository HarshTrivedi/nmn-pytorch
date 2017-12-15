import torch
import torch.nn.functional as F
import numpy as np

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)


class LSTMModel(torch.nn.Module):

    def __init__(self, question_word_embeddings_np, embedding_dim, D_map,
                 hidden_dim, vocab_size, D_ans_choices):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.question_word_embeddings = torch.nn.Embedding(
            vocab_size, embedding_dim)
        self.question_word_embeddings.weight = torch.nn.Parameter(
            torch.from_numpy(question_word_embeddings_np))
        self.question_word_embeddings.weight.requires_grad = False
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()
        self.map_to_answer_choices = torch.nn.Linear(D_map, D_ans_choices)

    def init_hidden(self):
        if use_gpu:
            return (torch.autograd.Variable(
                torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    torch.autograd.Variable(
                        torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input_token_ids_sequence):
        embeds = self.question_word_embeddings(input_token_ids_sequence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(input_token_ids_sequence.size()[1], 1, -1), self.hidden)
        softmax = torch.nn.Softmax()
        scores = self.map_to_answer_choices(self.hidden[0].view(1, -1))
        # return softmax(scores)
        return scores
