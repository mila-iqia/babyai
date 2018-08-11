import multiprocessing as mp
import numpy

def recv_conns(conns):
    """Receives the data coming from all the connections."""

    data = []
    wconns = mp.connection.wait(conns, timeout=.0)
    while len(wconns) > 0:
        for r in wconns:
            data.append(r.recv())
        wconns = mp.connection.wait(conns, timeout=.0)
    return data

class MultiEnvHead:
    """The head of several multi-environments.

    It communicates with these multi-environments through pipes: it first
    receives (env_id, return) tuples from them, updates the distribution
    over their environments (with a DistComputer object) after some time
    and sends them this distribution.

    This class enables to execute several multi-environments in different
    processes."""

    def __init__(self, num_menvs, num_envs, compute_dist=None):
        self.num_menvs = num_menvs
        self.num_envs = num_envs
        self.compute_dist = compute_dist

        self._init_connections()
        self.returns = {env_id: [0] * num_menvs for env_id in range(self.num_envs)}
        self.dist = numpy.ones((self.num_envs)) / self.num_envs
        self.update_dist()

    def _init_connections(self):
        self.locals, self.remotes = zip(*[mp.Pipe() for _ in range(self.num_menvs)])

    def _trim_returns(self):
        for env_id in self.returns:
            self.returns[env_id] = self.returns[env_id][-self.num_menvs:]

    def _recv_returns(self):
        data = recv_conns(self.locals)
        for env_id, returnn in data:
            self.returns[env_id].append(returnn)

    def _synthesize_returns(self):
        self.synthesized_returns = {}
        for env_id, returnn in self.returns.items():
            if len(returnn) > 0:
                self.synthesized_returns[env_id] = numpy.mean(returnn)

    def _send_dist(self):
        for local in self.locals:
            local.send(self.dist)

    def update_dist(self):
        self._recv_returns()
        self._synthesize_returns()
        self._trim_returns()

        if self.compute_dist is not None:
            self.dist = self.compute_dist(self.synthesized_returns)

        self._send_dist()

class MultiEnv:
    """A multi-environment.

    It simulates several environments: it first receives a distribution
    from its head, samples an environment from it, simulates it and
    then sends a (env_id, return) tuple to its head."""

    def __init__(self, envs, head_conn, seed=None):
        self.envs = envs
        self.head_conn = head_conn
        self.rng = numpy.random.RandomState(seed)

        self.num_envs = len(envs)
        self.returnn = None
        self.reset()

    def __getattr__(self, key):
        return getattr(self.env, key)

    def _recv_dist(self):
        data = recv_conns([self.head_conn])
        if len(data) > 0:
            self.dist = data[-1]

    def _select_env(self):
        self._recv_dist()
        self.env_id = self.rng.choice(range(self.num_envs), p=self.dist)
        self.env = self.envs[self.env_id]

    def _send_return(self):
        if self.returnn is not None:
            self.head_conn.send((self.env_id, self.returnn))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def reset(self):
        self._send_return()
        self.returnn = 0
        self._select_env()
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)
