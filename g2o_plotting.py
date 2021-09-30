import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

with open('g2o_greedy_results.json', 'r') as f:
    plot_dict = json.load(f)


for env in plot_dict:
    res = plot_dict[env]['agent_results']
    mean_percents = []
    std_percents = []
    mean_times = []
    std_times = []
    for agent in res.keys():
        # Get the data
        stats = res[agent]
        num_steps = stats['num_steps']
        max_steps = stats['max_steps']
        times = stats['timesteps']

        # Compute the percents/times
        percents = np.array([(num_steps[i] + 1) / max_steps[i] for i in range(len(num_steps))]) * 100
        step_times = np.array([times[i] / (num_steps[i] + 1) for i in range(len(num_steps))])

        # Compute the relevant stats
        mean_percent = np.mean(percents)
        std_percent = np.std(percents)
        mean_time = np.mean(step_times)
        std_time = np.std(step_times)

        # Append to our lists
        mean_percents.append(mean_percent)
        std_percents.append(std_percent)
        mean_times.append(mean_time)
        std_times.append(std_time)

    width = 0.35
    x_axis = np.arange(4)

    fig, axs1 = plt.subplots()

    # Plot the percentages
    axs1.bar(x_axis-width/2, mean_percents, width=width, color='g')
    axs1.errorbar(x_axis-width/2, mean_percents, yerr=std_percents, fmt='o', color='black')
    axs1.tick_params(axis='y', labelcolor='g')
    axs1.set_ylabel("Completion percentage")
    axs1.set_ylim(0, 105)
    # Plot the times on new axis
    axs2 = axs1.twinx()
    axs2.bar(x_axis+width/2, mean_times, width=width, color='r')
    axs2.errorbar(x_axis+width/2, mean_times, yerr=std_times, fmt='o', color='black')
    axs2.tick_params(axis='y', labelcolor='r')
    axs2.set_ylabel("Seconds per step")
    # Labels
    plt.title(f"Env: {env}")
    plt.xticks(x_axis, res.keys())

    plt.show()



