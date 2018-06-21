from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            action, task = data
            env_id, epoch_id = task
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset(env_id, epoch_id)
            
            conn.send((obs, env.env_id, env.epoch_id, reward, done, info))
        elif cmd == "reset":
            env_id, epoch_id = data
            obs = env.reset(env_id, epoch_id)
            conn.send((obs, env_id, epoch_id))
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
    
    def register(tasks):
        self.tasks = tasks
        self.len_tasks = len(tasks)
    
    def start(self):
        for local in self.locals:
            if len(self.tasks) > 0:
                local.send(("reset", self.tasks.pop()))
            else:
                local.send(("reset", (0,-1)))
        
        obss, env_ids, epoch_ids = [], [], []
        for local in self.locals:
            obs, env_id, epoch_id = local.recv()
            obss.append(obs)
            env_ids.append(env_id)
            epoch_ids.append(epoch_id)
        return obss, env_ids, epoch_ids
        
        #for local, env_id in zip(self.locals, env_ids):
        #    local.send(("reset", env_id))
        #results = [local.recv() for local in self.locals]
        #return results

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            if len(self.tasks) > 0:
                local.send(("step", (action, self.tasks.pop())))
            else:
                local.send(("step", (action, (0,-1))))
                
            #local.send(("step", (action, self.tasks)))
        
        obss, env_ids, epoch_ids, rewards, dones, infos = [], [], [], [], [], []
        for local in self.locals:
            obs, env_id, epoch_id, reward, done, info = local.recv()
            if not done and epoch_id >= 0:
                self.tasks.insert(0, (env_id, epoch_id))
            obss.append(obs)
            env_ids.append(env_id)
            epoch_ids.append(epoch_id)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        #results = zip(*[local.recv() for local in self.locals])
        #return results
        return obss, env_ids, epoch_ids, rewards, dones, infos

    def render(self):
        raise NotImplementedError

class MultiEnv:
    def __init__(self, envs):
        assert len(envs) > 0
        self.envs = envs
        self.env = None
        self.env_id = 0
        self.epoch_id = -1
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

    def reset(self, env_id, epoch_id):
        self._set_evn(env_id)
        self.epoch_id = epoch_id
        return self.env.reset()