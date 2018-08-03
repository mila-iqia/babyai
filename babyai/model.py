import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871

class AgentControllerFiLM(nn.Module):
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

class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out

class ImageBOWEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, reduce_fn=torch.mean):
        super(ImageBOWEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.reduce_fn = reduce_fn
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.reduce_fn(embeddings, dim=1)
        embeddings = torch.transpose(torch.transpose(embeddings, 1, 3), 2, 3)
        return embeddings

class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_instr=False, lang_model="gru", use_memory=False, arch="cnn1"):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model


        self.obs_space = obs_space

        # Define image embedding
        self.image_embedding_size = 128

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
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        elif arch == 'embcnn1':
            self.image_conv = nn.Sequential(
                ImageBOWEmbedding(obs_space["image"], embedding_dim=16, padding_idx=0, reduce_fn=torch.mean),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size, kernel_size=(3, 3)),
                nn.ReLU()
            )
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model == 'gru' or self.lang_model == 'conv' or self.lang_model == 'bigru':
                self.word_embedding_size = 128
                self.word_embedding = nn.Embedding(obs_space["instr"], self.word_embedding_size)
                if self.lang_model == 'gru' or self.lang_model == 'bigru':
                    self.instr_embedding_size = 128
                    instr_embedding_size = self.instr_embedding_size
                    if self.lang_model == 'bigru':
                        instr_embedding_size = instr_embedding_size // 2
                    self.instr_rnn = nn.GRU(self.word_embedding_size, instr_embedding_size, batch_first=True, bidirectional=(self.lang_model == 'bigru'))
                else:
                    kernel_dim = 64
                    kernel_sizes = [3,4]
                    self.instr_convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, self.word_embedding_size)) for K in kernel_sizes])
                    self.instr_embedding_size = kernel_dim * len(kernel_sizes)

            elif self.lang_model == 'bow':
                self.instr_embedding_size = 128
                hidden_units = [obs_space["instr"], 128, self.instr_embedding_size]
                layers = []
                for n_in, n_out in zip(hidden_units, hidden_units[1:]):
                    layers.append(nn.Linear(n_in, n_out))
                    layers.append(nn.ReLU())
                self.instr_bow = nn.Sequential(*layers)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and arch != "filmcnn" and not arch.startswith("expert_filmcnn"):
            self.embedding_size += self.instr_embedding_size

        if arch == "filmcnn":
            self.controller_1 = AgentControllerFiLM(in_features= self.instr_embedding_size, out_features= 64, in_channels = 3, imm_channels = 16)
            self.controller_2 = AgentControllerFiLM(in_features = self.instr_embedding_size, out_features= 64, in_channels = 32, imm_channels = 32)

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_')+1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module-1:
                    mod = ExpertControllerFiLM(in_features= self.instr_embedding_size, out_features= 128, in_channels = 128, imm_channels = 128)
                else:
                    mod = ExpertControllerFiLM(in_features = self.instr_embedding_size, out_features= self.image_embedding_size, in_channels = 128, imm_channels = 128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

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
            if type(embed_instr) == tuple:
                whole_context, embed_instr, mask_context = embed_instr

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if self.arch == "filmcnn":
            x = self.controller_1(x, embed_instr)
            x = self.image_conv_1(x)
            x = self.controller_2(x, embed_instr)
            x = self.image_conv_2(x)
        elif self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, embed_instr)
            x = F.relu(self.film_pool(x))
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

        if self.use_instr and self.arch != "filmcnn" and not self.arch.startswith("expert_filmcnn"):
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

        elif self.lang_model == 'bigru':
            self.instr_rnn.flatten_parameters()

            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, h_n = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, h_n = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            h_n = h_n.transpose(0,1).contiguous()
            h_n = h_n.view(h_n.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                hidden = h_n[iperm_idx]
            else:
                hidden = h_n

            if outputs.shape[1] < masks.shape[1]:
                masks = masks[:, :(outputs.shape[1]-masks.shape[1])] #The packing truncate the original length so we need to change mask to fit it

            return outputs, hidden, masks

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
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
