from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset(0)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset(data)
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].envs[0].observation_space
        self.action_space = self.envs[0].envs[0].action_space
        self.num_envs = len(self.envs[0].envs)
        self.num_procs = len(self.envs)

        self.locals = []
        for env in self.envs:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self, env_ids):
        for local, env_id in zip(self.locals, env_ids):
            local.send(("reset", env_id))
        results = [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(("step", action))
        results = zip(*[local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

class MultiEnv:
    def __init__(self, envs):
        assert len(envs) > 0
        self.envs = envs
        self.env = None
        self.env_id = 0
        self.num_envs = len(envs)
        #self.reset()
    
    def __getattr__(self, key):
        return getattr(self.env, key)
    
    def _set_evn(self, iid):
        assert iid >= 0 and iid < self.num_envs
        self.env = self.envs[iid]
        self.env_id = iid
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, id):
        self._set_evn(id)
        return self.env.reset()