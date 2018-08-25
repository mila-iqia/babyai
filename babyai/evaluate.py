import numpy as np
import gym

# Returns the performance of the agent on the environment for a particular number of episodes.
def evaluate(agent, env, episodes, model_agent=True, offsets=None):
    # Initialize logs
    if model_agent:
        agent.model.eval()
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    if offsets:
        count = 0

    for i in range(episodes):
        if offsets:
            # Ensuring test on seed offsets that generated successful demonstrations
            while count != offsets[i]:
                obs = env.reset()
                count += 1

        obs = env.reset()
        done = False

        num_frames = 0
        returnn = 0
        obss = []
        while not done:
            action = agent.act(obs)['action']
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward
            if num_frames > 30:
                break

        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    if model_agent:
        agent.model.train()
    return logs

# Function used for evaluation when using multiple processors
def evaluateProc(agent, penv, episodes, log_dict, env_names, proc_id, model_agent = True):

    # Initialize logs
    if model_agent:
        agent.model.eval()
    logs = {}
    for index in range(len(penv)):

        log_env = {"num_frames_per_episode": [], "return_per_episode": []}

        env = penv[index]
        for _ in range(episodes):
            obs = env.reset()
            done = False

            num_frames = 0
            returnn = 0
            obss = []
            while not(done):
                action = agent.act(obs)['action']
                action = action.item()
                obss.append(obs)
                obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)
                num_frames += 1
                returnn += reward
                if num_frames > 30:
                    break

            log_env["num_frames_per_episode"].append(num_frames)
            log_env["return_per_episode"].append(returnn)

        for key in log_env:
            log_env[key] = np.mean(log_env[key])
        logs[env_names[index][0]] = log_env

    if model_agent:
        agent.model.train()
    log_dict[proc_id] = logs
    

class ManyEnvs(gym.Env):

    def __init__(self, envs):
        self.envs = envs
        self.done = [False] * len(self.envs)

    def reset(self):
        many_obs = [env.reset() for env in self.envs]
        self.done = [False] * len(self.envs)
        return many_obs

    def step(self, actions):
        self.results = [env.step(action) if not done else self.last_results[i]
                        for i, (env, action, done)
                        in enumerate(zip(self.envs, actions, self.done))]
        self.done = [result[2] for result in self.results]
        self.last_results = self.results
        return zip(*self.results)

    def render(self):
        raise NotImplementedError


# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes):
    num_envs = 256

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        env.seed((int)(1e9) + 100 * seed + i)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {"num_frames_per_episode": [], 
            "return_per_episode": [], 
            "observations_per_episode": []}

    for i in range((episodes + num_envs - 1) // num_envs):
        many_obs = env.reset()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')
        while (num_frames == 0).any():
            action = agent.act_batch(many_obs)['action']
            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))

    return logs

