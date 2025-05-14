"""
Taxi-v3 Q-Learning Implementation
Introduction to Artificial Intelligence - Lab 6: Reinforcement Learning
Summer 2025

This script implements the Q-Learning algorithm to solve the Taxi-v3 environment
from Gymnasium library.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

def initialize_q_table(state_space, action_space):
    """
    Initialize the Q-table with zeros.
    
    Args:
        state_space (int): Size of the state space
        action_space (int): Size of the action space
        
    Returns:
        numpy.ndarray: Initialized Q-table with zeros
    """
    return np.zeros((state_space, action_space))

def choose_action(state, q_table, epsilon):
    """
    Choose an action using epsilon-greedy policy.
    
    Args:
        state (int): Current state
        q_table (numpy.ndarray): Q-table
        epsilon (float): Exploration rate
        
    Returns:
        int: Selected action
    """
    # Explore: choose a random action
    if np.random.random() < epsilon:
        return np.random.randint(0, q_table.shape[1])
    # Exploit: choose best action based on Q-values
    else:
        return np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Update the Q-table using the Q-Learning update rule.
    
    Args:
        q_table (numpy.ndarray): Q-table
        state (int): Current state
        action (int): Action taken
        reward (float): Reward received
        next_state (int): Next state
        alpha (float): Learning rate
        gamma (float): Discount factor
        
    Returns:
        numpy.ndarray: Updated Q-table
    """
    # Q-Learning update formula: Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    
    return q_table

def plot_metrics(episode_rewards, episode_lengths, window_size=100):
    """
    Plot training metrics.
    
    Args:
        episode_rewards (list): List of rewards for each episode
        episode_lengths (list): List of lengths for each episode
        window_size (int): Window size for moving average
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot episode rewards
    ax1.plot(episode_rewards, label='Episode Reward')
    # Calculate and plot moving average of rewards
    if len(episode_rewards) >= window_size:
        moving_avg_rewards = [np.mean(episode_rewards[i:i+window_size]) 
                             for i in range(len(episode_rewards) - window_size + 1)]
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg_rewards, 
                label=f'Moving Average ({window_size} episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(episode_lengths, label='Episode Length')
    # Calculate and plot moving average of lengths
    if len(episode_lengths) >= window_size:
        moving_avg_lengths = [np.mean(episode_lengths[i:i+window_size]) 
                             for i in range(len(episode_lengths) - window_size + 1)]
        ax2.plot(range(window_size-1, len(episode_lengths)), moving_avg_lengths, 
                label=f'Moving Average ({window_size} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('taxi_training_metrics.png')
    plt.show()

def render_episode(env, q_table, max_steps=100):
    """
    Render a single episode using the trained Q-table.
    
    Args:
        env (gym.Env): Gymnasium environment
        q_table (numpy.ndarray): Trained Q-table
        max_steps (int): Maximum steps per episode
    """
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # Render the environment
        clear_output(wait=True)
        env.render()
        time.sleep(0.5)  # Slow down rendering for better visualization
        
        # Choose the best action
        action = np.argmax(q_table[state])
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # Update state
        state = next_state
        
        # Check if episode is done
        if terminated or truncated:
            clear_output(wait=True)
            env.render()
            print(f"Episode finished after {step+1} steps with total reward {total_reward}")
            break
    
    env.close()

def train(env, num_episodes=50000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.9999, 
          epsilon_min=0.01, max_steps=200, print_interval=1000):
    """
    Train the agent using Q-Learning.
    
    Args:
        env (gym.Env): Gymnasium environment
        num_episodes (int): Number of training episodes
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon_start (float): Initial exploration rate
        epsilon_decay (float): Decay rate of epsilon after each episode
        epsilon_min (float): Minimum exploration rate
        max_steps (int): Maximum steps per episode
        print_interval (int): Interval to print progress
        
    Returns:
        tuple: Trained Q-table and training metrics
    """
    # Get state and action space dimensions
    state_space = env.observation_space.n
    action_space = env.action_space.n
    
    # Initialize Q-table
    q_table = initialize_q_table(state_space, action_space)
    
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    
    # Current epsilon value
    epsilon = epsilon_start
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        state, _ = env.reset()
        total_reward = 0
        
        # Episode loop
        for step in range(max_steps):
            # Choose action (epsilon-greedy)
            action = choose_action(state, q_table, epsilon)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Update Q-table
            q_table = update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
            
            # Update metrics
            total_reward += reward
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_length = np.mean(episode_lengths[-print_interval:])
            print(f"Episode: {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.2f}, Epsilon: {epsilon:.4f}")
    
    return q_table, episode_rewards, episode_lengths

def experiment_with_hyperparameters():
    """
    Run experiments with different hyperparameter settings.
    """
    # Hyperparameter configurations to test
    configs = [
        {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.9999, "name": "Default"},
        {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.9999, "name": "High Alpha"},
        {"alpha": 0.1, "gamma": 0.8, "epsilon_decay": 0.9999, "name": "Low Gamma"},
        {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.999, "name": "Fast Decay"}
    ]
    
    num_episodes = 50000  # Reduced for faster experiments
    results = {}
    
    for config in configs:
        print(f"\nRunning experiment with {config['name']} parameters:")
        print(f"Alpha: {config['alpha']}, Gamma: {config['gamma']}, Epsilon Decay: {config['epsilon_decay']}")
        
        env = gym.make("Taxi-v3")
        q_table, rewards, lengths = train(
            env, 
            num_episodes=num_episodes, 
            alpha=config['alpha'], 
            gamma=config['gamma'], 
            epsilon_decay=config['epsilon_decay'],
            print_interval=2000
        )
        env.close()
        
        # Save results
        results[config['name']] = {
            'rewards': rewards,
            'lengths': lengths,
            'final_avg_reward': np.mean(rewards[-100:]),
            'convergence_episode': np.argmax(
                [np.mean(rewards[i:i+100]) for i in range(len(rewards)-100)]
            )
        }
        
        print(f"Final Average Reward (last 100 episodes): {results[config['name']]['final_avg_reward']:.2f}")
        print(f"Approximate Convergence Episode: {results[config['name']]['convergence_episode']}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    window_size = 100
    
    for name, data in results.items():
        rewards = data['rewards']
        if len(rewards) >= window_size:
            moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards)-window_size+1)]
            plt.plot(range(window_size-1, len(rewards)), moving_avg, label=name)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (over 100 episodes)')
    plt.title('Performance Comparison of Different Hyperparameter Settings')
    plt.legend()
    plt.grid(True)
    plt.savefig('taxi_hyperparameter_comparison.png')
    plt.show()
    
    return results

def main():
    """
    Main function to run the training and visualization.
    """
    # Create Taxi environment
    env = gym.make("Taxi-v3")
    
    # Set hyperparameters
    num_episodes = 50000
    alpha = 0.1         # Learning rate
    gamma = 0.99        # Discount factor
    epsilon_start = 1.0 # Initial exploration rate
    epsilon_decay = 0.9999 # Epsilon decay rate
    epsilon_min = 0.01  # Minimum exploration rate
    
    print("Starting training...")
    start_time = time.time()
    
    # Train the agent
    q_table, episode_rewards, episode_lengths = train(
        env, 
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot metrics
    plot_metrics(episode_rewards, episode_lengths)
    
    # Visualize a trained episode
    try:
        env = gym.make("Taxi-v3", render_mode="human")
        print("Rendering a trained episode...")
        render_episode(env, q_table)
    except Exception as e:
        print(f"Could not render environment: {e}")
        print("If running in a non-interactive environment, rendering may not be supported.")
    finally:
        env.close()
    
    # Save Q-table
    np.save('taxi_q_table.npy', q_table)
    print("Q-table saved to 'taxi_q_table.npy'")
    
    # Experiment with different hyperparameters
    print("\nWould you like to run experiments with different hyperparameters? (y/n)")
    choice = input().lower()
    if choice == 'y':
        experiment_results = experiment_with_hyperparameters()
        print("\nExperiment results summary:")
        for name, data in experiment_results.items():
            print(f"{name}: Final Avg Reward = {data['final_avg_reward']:.2f}, "
                  f"Converged at Episode ≈ {data['convergence_episode']}")

if __name__ == "__main__":
    main()