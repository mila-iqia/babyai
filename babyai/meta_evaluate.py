# use botAgent metapolicy to decompose instructions
def evaluate_meta(agent, env, episodes, verbose=True):
    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    for _ in range(episodes):
        obs = env.reset()
        agent.on_reset(env)
        num_frames = 0
        returnn = 0
        obss = []

        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            num_frames += 1
            returnn += reward
            obss.append(obs)
        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)

        if verbose:
            if returnn == 0:
                print(returnn)
                print(obs['mission'])
                print(agent.bot.stack)
                print(env)
                print("")
    return logs
