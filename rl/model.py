import torch
import torch.nn as nn
import torch.nn.functional as F


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy

class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 4 * 4, 512)

        self.critic_linear = nn.Linear(512, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 4 * 4)
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), x

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ObsNorm(nn.Module):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        super(ObsNorm, self).__init__()
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.register_buffer('count', torch.zeros(1).double() + 1e-2)
        self.register_buffer('sum', torch.zeros(shape).double())
        self.register_buffer('sum_sqr', torch.zeros(shape).double() + 1e-2)

        self.register_buffer('mean', torch.zeros(shape),)
        self.register_buffer('std', torch.ones(shape))

    def update(self, x):
        self.count += x.size(0)
        self.sum += x.sum(0, keepdim=True).double()
        self.sum_sqr += x.pow(2).sum(0, keepdim=True).double()

        self.mean = self.sum / self.count
        self.std = (self.sum_sqr / self.count - self.mean.pow(2)).clamp(1e-2, 1e9).sqrt()

        self.mean = self.mean.float()
        self.std = self.std.float()

    def __call__(self, x):
        if self.demean:
            x = x - self.mean
        if self.destd:
            x = x / self.std
        if self.clip:
            x = x.clamp(-self.clip, self.clip)
        return x


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x)
        probs = F.softmax(x)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        LAYER_SIZE = 64

        self.a_fc1 = nn.Linear(num_inputs, LAYER_SIZE)
        self.a_fc2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)

        self.v_fc1 = nn.Linear(num_inputs, LAYER_SIZE)
        self.v_fc2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.v_fc3 = nn.Linear(LAYER_SIZE, 1)

        num_outputs = action_space.n
        self.dist = Categorical(LAYER_SIZE, num_outputs)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = self.v_fc1(inputs)
        x = F.relu(x)

        x = self.v_fc2(x)
        x = F.relu(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.relu(x)

        x = self.a_fc2(x)
        x = F.relu(x)

        return value, x
