import gym
env = gym.make('HalfCheetah-v2')
env = gym.make('CartPole-v0')
print(env.action_space.n)
print(env.observation_space.shape)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()