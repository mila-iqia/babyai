import numpy as np

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

