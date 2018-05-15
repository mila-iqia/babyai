import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_instr, use_memory, use_cnn):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.use_cnn = use_cnn

        # Define image embedding
        self.image_embedding_size = 64
        if not self.use_cnn:
            self.image_fc = nn.Linear(obs_space["image"], self.image_embedding_size)
        else:
            self.image_fc = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16,
                                                    kernel_size=(2, 2)),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                          nn.Conv2d(in_channels=16, out_channels=32,
                                                    kernel_size=(2, 2)),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size,
                                                    kernel_size=(2, 2)),
                                          nn.ReLU()
                                          )

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define instruction embedding
        if self.use_instr:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["instr"], self.word_embedding_size)
            self.instr_embedding_size = 128
            self.instr_rnn = nn.GRU(self.word_embedding_size, self.instr_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr:
            self.embedding_size += self.instr_embedding_size

        # Define actor's model
        self.a_fc1 = nn.Linear(self.embedding_size, 64)
        self.a_fc2 = nn.Linear(64, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(self.embedding_size, 64)
        self.c_fc2 = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        if self.use_instr:
            embed_instr = self._get_embed_instr(obs.instr)

        if not self.use_cnn:
            obs.image = obs.image.reshape(obs.image.shape[0], -1)
        else:
            obs.image = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        x = self.image_fc(obs.image)

        if self.use_cnn:
            x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr:
            embedding = torch.cat((embedding, embed_instr), dim=1)

        x = self.a_fc1(embedding)
        x = F.tanh(x)
        x = self.a_fc2(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.c_fc1(embedding)
        x = F.tanh(x)
        x = self.c_fc2(x)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_instr(self, instr):
        self.instr_rnn.flatten_parameters()
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]