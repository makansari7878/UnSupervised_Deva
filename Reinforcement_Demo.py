import random

class SaimpleEnvironment:

    def __init__(self):
        self.steps = 12

    def get_observation(self):
        return [0.0,0.0,0.0]

    def get_actions(self):
        return [0,1]

    def is_done(self):
       return self.steps == 0

    def action(self, action:int):
        if self.is_done():
            raise Exception("GAME IS OVER")
        else:
            self.steps -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env : SaimpleEnvironment):
        current_state = env.get_observation()
        print("your state :", current_state)

        actions = env.get_actions()
        rewards = env.action(random.choice(actions))
        self.total_reward += rewards
        print("agent rewards are :", self.total_reward)

env = SaimpleEnvironment()
agent = Agent()
agent.step(env)

steps = 0

while not env.is_done():
    steps = steps + 1
    agent.step(env)

print(f"at step {steps} agent is get total reward {agent.total_reward}")




