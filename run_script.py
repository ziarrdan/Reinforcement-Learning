import matplotlib.pyplot as plt
import numpy as np
from agent import Dqn
import pandas as pd
import gym


def run_episode(env, agent, epsilon, final_epsilon, epsilon_decay, episode_length, episode, scores, steps, training=True):
    should_end = False
    num_steps = 0
    state = env.reset()
    score = 0
    for t in range(episode_length):
        action = agent.take_action(state, epsilon)
        state_next, reward, done, _ = env.step(action)
        if training:
            agent.learn(state, action, state_next, reward, done)
        state = state_next
        score += reward
        num_steps = t
        if done:
            break

    scores.append(score)
    steps.append(num_steps)
    epsilon = max(final_epsilon, epsilon * epsilon_decay)
    if len(scores) > 100:
        mean_100 = np.mean(scores[-100:])
    else:
        mean_100 = np.mean(scores)
    print('Score at episode ' + str(episode) + ': ', score)
    print('Average score of last 100 episodes at episode ' + str(episode) + ': ', str(mean_100))

    if training and np.mean(mean_100) > 200:
        print('Solved with average score of ' + str(mean_100) + ' for the last 100 episodes!')
        should_end = True

    return should_end, epsilon


def run(gamma=0.99, alpha=0.0005, network_struct=(50,50), c=10, testing=False):
    scores = []
    steps = []
    test_scores = []
    test_steps = []
    epsilon = 0.99
    epsilon_decay = 0.995
    final_epsilon = 0
    num_episodes = 1000
    episode_length = 1000
    episode = 0
    env = gym.make('LunarLander-v2')
    env.seed(123456789)
    agent = Dqn(gamma=gamma, alpha=alpha, network_struct=network_struct, c=c)

    while episode < num_episodes:
        last_episode, epsilon = run_episode(env, agent, epsilon, final_epsilon, epsilon_decay,
                                            episode_length, episode, scores, steps)
        episode += 1
        if last_episode:
            break

    if testing:
        for i in range(100):
            last_episode, _ = run_episode(env, agent, 0, 0, 0, episode_length, 0, test_scores, test_steps, False)

            if last_episode:
                break

    return scores, steps, episode, test_scores


def gen_fig_1_and_2():
    scores, steps, episode, test_scores = run(testing=True)

    plt.plot(range(episode), steps)
    steps_df = pd.DataFrame(steps)
    steps_rolling_mean = steps_df.rolling(window=25).mean()
    plt.plot(range(25, episode), steps_rolling_mean[25:].to_numpy(), label='Rolling Mean')
    plt.ylabel('Steps', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Number of steps per Each Training Episode', fontsize=12)
    plt.savefig('Figures/steps_training.png')
    plt.show()

    plt.plot(range(episode), scores)
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Reward for Each Training Episode', fontsize=12)
    plt.savefig('Figures/reward_training.png')
    plt.show()

    plt.plot(range(100), test_scores)
    plt.axhline(y=np.mean(test_scores), color='r')
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Reward for Each Episode using Trained Agent', fontsize=12)
    plt.savefig('Figures/reward_testing.png')
    plt.show()


def gen_fig_3_1():
    plot_steps = []
    plot_scores = []
    plot_episodes = []
    gammas = [0.90, 0.95, 0.99]

    for gamma in gammas:
        scores, steps, episode, test_scores = run(gamma=gamma)

        plot_episodes.append(episode)
        plot_steps.append(steps)
        plot_scores.append(scores)

    for i in range(len(gammas)):
        #plt.plot(range(plot_episodes[i]), plot_steps[i], label='γ=' + str(gammas[i]))
        steps_df = pd.DataFrame(plot_steps[i])
        steps_rolling_mean = steps_df.rolling(window=25).mean()
        plt.plot(range(25, plot_episodes[i]), steps_rolling_mean[25:].to_numpy(), label='γ=' + str(gammas[i]))
    plt.legend()
    plt.ylabel('Steps', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Number of steps per Each Training Episode', fontsize=12)
    plt.savefig('Figures/steps_for_gammas.png')
    plt.show()

    for i in range(len(gammas)):
        plt.plot(range(plot_episodes[i]), plot_scores[i], label='γ=' + str(gammas[i]))
    plt.legend()
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Reward for Each Training Episode', fontsize=12)
    plt.savefig('Figures/reward_training_for_gammas.png')
    plt.show()


def gen_fig_3_2():
    plot_steps = []
    plot_scores = []
    plot_episodes = []
    alphas = [0.0001, 0.0005, 0.005]

    for alpha in alphas:
        scores, steps, episode, test_scores = run(alpha=alpha)

        plot_episodes.append(episode)
        plot_steps.append(steps)
        plot_scores.append(scores)

    for i in range(len(alphas)):
        #plt.plot(range(plot_episodes[i]), plot_steps[i], label='α=' + str(alphas[i]))
        steps_df = pd.DataFrame(plot_steps[i])
        steps_rolling_mean = steps_df.rolling(window=25).mean()
        plt.plot(range(25, plot_episodes[i]), steps_rolling_mean[25:].to_numpy(), label='α=' + str(alphas[i]))
    plt.legend()
    plt.ylabel('Steps', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Number of steps per Each Training Episode', fontsize=12)
    plt.savefig('Figures/steps_for_alphas.png')
    plt.show()

    for i in range(len(alphas)):
        plt.plot(range(plot_episodes[i]), plot_scores[i], label='α=' + str(alphas[i]))
    plt.legend()
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Reward for Each Training Episode', fontsize=12)
    plt.savefig('Figures/reward_training_for_alphas.png')
    plt.show()


def gen_fig_3_3():
    plot_steps = []
    plot_scores = []
    plot_episodes = []
    nets = [(10, 10), (50, 50), (100, 100)]
    nets_str = ['10x10', '50x50', '100x100']

    for net in nets:
        scores, steps, episode, test_scores = run(network_struct=net)

        plot_episodes.append(episode)
        plot_steps.append(steps)
        plot_scores.append(scores)

    for i in range(len(nets)):
        #plt.plot(range(plot_episodes[i]), plot_steps[i], label='ANN=' + str(nets_str[i]))
        steps_df = pd.DataFrame(plot_steps[i])
        steps_rolling_mean = steps_df.rolling(window=25).mean()
        plt.plot(range(25, plot_episodes[i]), steps_rolling_mean[25:].to_numpy(), label='ANN=' + str(nets_str[i]))
    plt.legend()
    plt.ylabel('Steps', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Number of steps per Each Training Episode', fontsize=12)
    plt.savefig('Figures/steps_for_nets.png')
    plt.show()

    for i in range(len(nets)):
        plt.plot(range(plot_episodes[i]), plot_scores[i], label='ANN=' + str(nets_str[i]))
    plt.legend()
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Reward for Each Training Episode', fontsize=12)
    plt.savefig('Figures/reward_training_for_nets.png')
    plt.show()


def gen_fig_3_4():
    plot_steps = []
    plot_scores = []
    plot_episodes = []
    Cs = [1, 10, 100]

    for c in Cs:
        scores, steps, episode, test_scores = run(c=c)

        plot_episodes.append(episode)
        plot_steps.append(steps)
        plot_scores.append(scores)

    for i in range(len(Cs)):
        #plt.plot(range(plot_episodes[i]), plot_steps[i], label='α=' + str(alphas[i]))
        steps_df = pd.DataFrame(plot_steps[i])
        steps_rolling_mean = steps_df.rolling(window=25).mean()
        plt.plot(range(25, plot_episodes[i]), steps_rolling_mean[25:].to_numpy(), label='c=' + str(Cs[i]))
    plt.legend()
    plt.ylabel('Steps', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Number of steps per Each Training Episode', fontsize=12)
    plt.savefig('Figures/steps_for_cs.png')
    plt.show()

    for i in range(len(Cs)):
        plt.plot(range(plot_episodes[i]), plot_scores[i], label='c=' + str(Cs[i]))
    plt.legend()
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title('Reward for Each Training Episode', fontsize=12)
    plt.savefig('Figures/reward_training_for_cs.png')
    plt.show()


def main():
    gen_fig_1_and_2()
    gen_fig_3_1()
    gen_fig_3_2()
    gen_fig_3_3()
    gen_fig_3_4()

if __name__ == "__main__":
    main()