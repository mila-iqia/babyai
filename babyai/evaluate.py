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
            action = agent.get_action([obs])
            action = action.item()
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward

        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)

    return logs

'''
def evaluateProc(agent, env, env_ids):
    
    assert len(env_ids) == env.num_procs
    
    obs = env.reset(env_ids)
    done = [False] * env.num_procs
    stopUpdating = [False] * env.num_procs
    
    num_frames = [0] * env.num_procs
    returnn = [0] * env.num_procs
    obss = [[]] * env.num_procs
    
    while not all(stopUpdating):
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done, with_procs=True)
        for id in range(env.num_procs):
            
            if not stopUpdating[id]:
                num_frames[id] += 1
                returnn[id] += reward[id]
                obss[id].append(obs[id])
            
            if done[id] and not stopUpdating[id]:
                stopUpdating[id] = True
    
    return num_frames, returnn, obss
'''

def evaluateProc(agent, env, num_epochs):
    
    obs, pre_env_id, pre_epoch_id = env.start()
    totalEpochDone = 0
    num_frames = np.zeros((env.num_envs, num_epochs), dtype='float32')
    returnn = np.zeros((env.num_envs, num_epochs), dtype='float32')
    finished = np.zeros((env.num_envs, num_epochs), dtype='bool')
    
    while not finished.all():
        action = agent.get_action(obs)
        obs, env_id, epoch_id, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done, with_procs=True)
        
        for id in range(env.num_procs):
            enid = pre_env_id[id]
            epid = pre_epoch_id[id]
            if epid >= 0:
                num_frames[enid, epid] += 1.
                returnn[enid, epid] += reward[id]
            if done[id]:
                finished[enid, epid] = True
        pre_env_id = env_id
        pre_epoch_id = epoch_id
    
    return num_frames.mean(1), returnn.mean(1)