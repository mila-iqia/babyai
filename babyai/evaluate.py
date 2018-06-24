import numpy as np

# Returns the performance of the agent on the environment for a particular number of episodes.
def evaluate(agent, env, episodes):
    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    for _ in range(episodes):
        obs = env.reset()
        done = False

        num_frames = 0
        returnn = 0
        obss = []
        while not(done):
            action = agent.get_action(obs)
            action = action.item()
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

    return logs

def evaluateProc(agent, penv, episodes, log_dict, env_names, proc_id):

    # Initialize logs
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
                action = agent.get_action(obs)
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

    log_dict[proc_id] = logs

