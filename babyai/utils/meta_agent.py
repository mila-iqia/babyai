from babyai.stackbot import StackBot


class HandcraftedMetacontroller:
    def __init__(self, env, agent):
        """
        get meta-policy from StackBot
        control policy trained by AllControllerTasks environment
        """
        self.agent = agent
        self.agent.model.eval()
        self.on_reset(env)

    def on_reset(self, env):
        'reset bot, agent and last action'
        self.env = env
        self.action = None
        self.bot = StackBot(self.env)
        self.agent.on_reset()
        self.instr = None

    def get_agent_action(self, obs):
        'plan what action to get from the agent'
        # find instruction, if still none, do bot's action
        if self.instr is None:
            self.instr = self.bot.get_instruction()
        if self.instr is None:
            return self.action
        # use instruction to plan next action
        obs['mission'] = self.instr
        action = self.agent.act(obs)['action']
        # if last in subtask, remove instruction
        if action == self.env.actions.done:
            self.instr = None
        return action

    def get_action(self, obs):
        'get next action'
        # replan and get bot's action
        self.action = self.bot.replan(self.action)
        if not self.bot.stack:
            return self.env.actions.done
        # get action from instruction on stack
        action = self.get_agent_action(obs)
        while action is None:
            action = self.get_agent_action(obs)
        # update action for next time
        self.action = action
        return action
