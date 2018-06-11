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

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class Controller(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=imm_channels, out_channels=64, kernel_size=(1,1)),
            nn.ReLU()
        )
        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        return self.conv(x) * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)



class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_instr=False, lang_model="gru", use_memory=False, arch="cnn1"):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model

        
        self.obs_space = obs_space

        # Define image embedding
        self.image_embedding_size = 64

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch == "cnn2":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
                nn.Conv2d(in_channels=16, out_channels=self.image_embedding_size, kernel_size=(3, 3)),
                nn.ReLU()
            )
        elif arch == "filmcnn":
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")
            
            self.image_conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.image_conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size, kernel_size=(2, 2)),
                nn.ReLU()
            )

        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model == 'gru' or self.lang_model == 'conv':
                self.word_embedding_size = 32
                self.word_embedding = nn.Embedding(obs_space["instr"], self.word_embedding_size)
                if self.lang_model == 'gru':
                    self.instr_embedding_size = 128
                    self.instr_rnn = nn.GRU(self.word_embedding_size, self.instr_embedding_size, batch_first=True)
                else:
                    kernel_dim = 64
                    kernel_sizes = [3,4]
                    self.instr_convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, self.word_embedding_size)) for K in kernel_sizes])
                    self.instr_embedding_size = kernel_dim * len(kernel_sizes)
                    
            elif self.lang_model == 'bow':
                self.instr_embedding_size = 128
                hidden_units = [obs_space["instr"], 64, self.instr_embedding_size]
                layers = []
                for n_in, n_out in zip(hidden_units, hidden_units[1:]):
                    layers.append(nn.Linear(n_in, n_out))
                    layers.append(nn.ReLU())
                self.instr_bow = nn.Sequential(*layers)



        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and arch != "filmcnn":
            self.embedding_size += self.instr_embedding_size

    
        if arch == "filmcnn":
            self.controller_1 = Controller(in_features= self.instr_embedding_size, out_features= 64, in_channels = 3, imm_channels = 16)
            self.controller_2 = Controller(in_features = self.instr_embedding_size, out_features= 64, in_channels = 32, imm_channels = 32)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

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

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        
        if self.arch == "filmcnn":
            x = self.controller_1(x, embed_instr)
            x = self.image_conv_1(x)
            x = self.controller_2(x, embed_instr)
            x = self.image_conv_2(x)
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and self.arch != "filmcnn":
            embedding = torch.cat((embedding, embed_instr), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_instr(self, instr):
        if self.lang_model == 'gru':
            self.instr_rnn.flatten_parameters()
            _, hidden = self.instr_rnn(self.word_embedding(instr))
            return hidden[-1]

        elif self.lang_model == 'conv':
            inputs = self.word_embedding(instr).unsqueeze(1) # (B,1,T,D)
            inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.instr_convs]
            inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]

            return torch.cat(inputs, 1)
        elif self.lang_model == 'bow':
            #self.instr_bow.flatten_parameters()
            device = torch.device("cuda" if instr.is_cuda else "cpu")
            input_dim = self.obs_space["instr"]
            input = torch.zeros((instr.size(0), input_dim), device=device)
            idx = torch.arange(instr.size(0), dtype=torch.int64)
            input[idx.unsqueeze(1), instr] = 1.
            return self.instr_bow(input)
        else:
            ValueError("Undefined instruction architecture: {}".format(self.lang_model))
            
