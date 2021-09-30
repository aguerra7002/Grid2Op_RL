# Bringing in an example agent
import grid2op
from grid2op.Agent import DoNothingAgent, RecoPowerlineAgent, TopologyGreedy, PowerLineSwitch
from tqdm.notebook import tqdm  # for easy progress bar
import json
import datetime
import os

# Run the standard gym loop, saving the observations
RESULTS_FILENAME = 'g2o_greedy_results.json'

# List of all environments we can run on.
env_list = grid2op.list_available_remote_env()

# Final dict where we will store all our results
final_dict = {}

# This code simply eliminates all the environments for which we already have results for.
if os.path.isfile(RESULTS_FILENAME):
    with open(RESULTS_FILENAME, 'r') as f:
        final_dict = json.load(f)
    # Get rid of all environments which
    env_list = list(set(env_list) - set(final_dict.keys()))

# How many simuls do we do for each agent on each environment
N_SIMULS = 5

for env_name in env_list:
    print(f"{env_name}:")
    env = grid2op.make(env_name)
    agents_dict = {
                    "do_nothing": DoNothingAgent(env.action_space),
                    "reco_powerline": RecoPowerlineAgent(env.action_space),
                    #"topo_greedy": TopologyGreedy(env.action_space),
                    "power_switch": PowerLineSwitch(env.action_space)
                   }
    agents_results = {}
    for agent_name in agents_dict.keys():
        print(f"\t{agent_name}")
        agent = agents_dict[agent_name]
        nb_steps = []
        mx_steps = []
        times = []
        for simul in range(N_SIMULS):
            print(simul)
            start_time = datetime.datetime.now()
            env.seed(simul)
            obs = env.reset()
            reward = env.reward_range[0]
            done = False
            nb_step = 0
            with tqdm(total=env.chronics_handler.max_timestep(), disable=False) as pbar:
                while True:
                    action = agent.act(obs, reward, done)
                    obs, reward, done, _ = env.step(action)
                    pbar.update(1)
                    if done:
                        seconds = (datetime.datetime.now() - start_time).total_seconds()
                        times.append(seconds)
                        nb_steps.append(nb_step)
                        mx_steps.append(env.chronics_handler.max_timestep())
                        break
                    nb_step += 1
        # When all the simulations are done, we save the results to the agent dict
        agents_results[agent_name] = {"num_steps": nb_steps, "max_steps": mx_steps, "timesteps": times}
    # Now that we have run all agents for <num_simuls> we save the results
    final_dict[env_name] = {"agent_results": agents_results}
    # Save the file after every env is finished running
    with open(RESULTS_FILENAME, 'w') as f:
        json.dump(final_dict, f)
