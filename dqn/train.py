import gymnasium as gym
import numpy as np
from util.gymnastics import epsilon_gen, plot_scores, init_random
from dqn.agent import Agent
from dqn.simulate import gym_simulate
from gymnasium.wrappers import AtariPreprocessing


def start_episode(env: gym.Env):
    """Method to call to start a new episode for pong in DQN training."""
    state, _ = env.reset()
    obs, _, _, _, _ = env.step(1)  # Starts the game :)
    return np.stack([state, obs])


def train(env: gym.Env, agent: Agent, max_timesteps=int(1e6)) -> list[int]:
    # Records all episode scores.
    scores = []
    # Tracks the current episode score.
    score = 0.0
    # Tracks the current episode number.
    n_episode = 1
    # Create the epsilon generator. Start: 0.1, decay: 0.995, min: 0.01.
    eps_gen = epsilon_gen(eps_start=0.1, eps_decay=0.995, eps_min=0.01)
    # Get the next epsilon.
    epsilon = next(eps_gen)
    # Get the first state.
    state = start_episode(env)
    # Run DQN training for max_timesteps.
    for t in range(max_timesteps):
        # Select an action calling agent.act passing state and epsilon.
        action = agent.act(state, epsilon)
        # Make a step in the environment with the selection action.
        observation, reward, terminated, truncated, _ = env.step(action)
        # TODO: Build the new state (i.e., two stacked frames).
        next_state = np.stack([state[1], observation])
        # Determines if the episode ended
        done = terminated or truncated
        # Call agent.step with (state, action, reward, next_state, done). That will take care
        #       of collecting experiences and learning - we'll see later :)
        agent.step(state, action, reward, next_state, done)
        # Prepares for the next iteration, updating score and state, and checking for completion.
        state = next_state
        score += reward
        if done:  # Ends the episode
            scores.append(score)
            avg = np.mean(scores[-25:])
            print(
                f"\rEpisode {n_episode}\tScore: {score:.2f}\tT={t:6} (avg={avg:.2f})",
                end="\n" if n_episode % 25 == 0 else "",
            )
            score = 0.0
            n_episode += 1
            epsilon = next(eps_gen)
            state = start_episode(env)

    # Checkpoint the agent at the end of training, also save the scores for plotting.
    agent.checkpoint()
    np.savetxt("dqn_scores.csv", np.asarray(scores, dtype=np.int16), delimiter=",")
    return scores


def pretrained_simulation():
    pretrained_agent = Agent(preload_file="solution/dqn_weights_pre.pth")
    pretrained_scores = np.loadtxt(f"solution/dqn_scores_pre.csv", delimiter=",").astype(np.int16)
    plot_scores(pretrained_scores)
    return gym_simulate(pretrained_agent)


def train_agent():
    with gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0) as env:
        env = init_random(env)
        env = AtariPreprocessing(env)
        agent = Agent(action_size=env.action_space.n)
        scores = train(env, agent)
    plot_scores(scores)
    gym_simulate(agent)
