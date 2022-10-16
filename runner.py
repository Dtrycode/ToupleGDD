import numpy as np
from itertools import count
import torch
import rl_agents
import models
import statistics
from tqdm import tqdm
import os
import time
from statistics import mean

torch.manual_seed(123)
np.random.seed(123)

class Runner:
    ''' run an agent in an environment '''
    def __init__(self, train_env, test_env, agent, training):
        self.train_env = train_env
        self.test_env = test_env # environment for testing
        self.agent = agent
        self.training = training

    def play_game(self, num_iterations, epsilon, training=True, time_usage=False, one_time=False):
        ''' play the game num_iterations times 
        Arguments:
            time_usage: off if False; True: print average time usage for seed set generation
            one_time: generate the seed set at once without regenerating embeddings
        '''
        if training:
            self.env = self.train_env
        else:
            self.env = self.test_env

        c_rewards = []
        im_seeds = []

        if time_usage:
            total_time = 0.0 # total time for all iterations on all testing graphs

        for iteration in range(num_iterations):
            # handle multiple graphs for evaluation during training
            if training:
                self.env.reset()

                for i in count():
                    state = torch.tensor(self.env.state, dtype=torch.long)
                    action = self.agent.select_action(self.env.graph, state, epsilon, training=training).item()
                    reward, done = self.env.step(action)
                    # this game is over
                    if done:
                        # memorize the trajectory
                        self.agent.memorize(self.env)
                        break
            else:
                for g_idx in range(len(self.env.graphs)):
                    # measure time of generating initial embedding if need to print time
                    # this may prevent the initial embedding generation of rl_agent side:
                    #   the number of deep walk training iterations
                    if time_usage and (id(self.env.graphs[g_idx]) not in self.agent.graph_node_embed):
                        start_time = time.time()
                        self.agent.graph_node_embed[id(self.env.graphs[g_idx])] = models.get_init_node_embed(self.env.graphs[g_idx], 0, self.agent.device) # epochs for initial embedding
                        print(f'Time of generating initial embedding for {self.env.graphs[g_idx].path_graph}: {time.time()-start_time:.2f} seconds')
                        
                    if time_usage:
                        start_time = time.time()
                        time_reward = [0.0] # time of calculating reward, needs to be subtracted
                    else:
                        time_reward = None

                    self.env.reset(g_idx, training=training)
                    if one_time:
                        state = torch.tensor(self.env.state, dtype=torch.long)
                        actions = self.agent.select_action(self.env.graph, state, epsilon, training=training, budget=self.env.budget).tolist()

                        # no sort of actions selected
                        im_seeds.append(actions)

                        if time_usage:
                            total_time += time.time() - start_time

                        final_reward = self.env.compute_reward(actions)
                        c_rewards.append(final_reward)

                    else:
                        for i in count():
                            state = torch.tensor(self.env.state, dtype=torch.long)
                            action = self.agent.select_action(self.env.graph, state, epsilon, training=training).item()

                            final_reward, done = self.env.step(action, time_reward)
                            # this game is over
                            if done:
                                # no sort of action selected
                                im_seeds.append(self.env.actions)
                                c_rewards.append(final_reward)
                                break
                        if time_usage:
                            total_time += time.time() - start_time - time_reward[0]
        if time_usage:
            print(f'Seed set generation per iteration time usage is: {total_time/num_iterations:.2f} seconds')
        return c_rewards, im_seeds


    def train(self, num_epoch, model_file, result_file):
        ''' let agent act and learn from the environment '''
        # pretrain
        tqdm.write('Pretraining:')
        self.play_game(1000, 1.0)

        eps_start = 1.0
        eps_end = 0.05
        eps_step = 10000.0
        # train
        tqdm.write('Starting fitting:')
        progress_fitting = tqdm(total=num_epoch)
        for epoch in range(num_epoch):
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - epoch) / eps_step)
            
            if epoch % 10 == 0:
                self.play_game(10, eps)

            if epoch % 10 == 0:
                # test
                rewards, seeds = self.play_game(1, 0.0, training=False)
                tqdm.write(f'{epoch}/{num_epoch}: ({str(seeds[0])[1:-1]}) | {rewards[0]}')                

            if epoch % 10 == 0:
                # save model
                self.agent.save_model(model_file + str(epoch))

            if epoch % 100 == 0:
                self.agent.update_target_net()
            # train the model
            self.agent.fit()

            progress_fitting.update(1)

        # show test results after training
        rewards, seeds = self.play_game(1, 0.0, training=False)
        tqdm.write(f'{num_epoch}/{num_epoch}: ({str(seeds[0])[1:-1]}) | {rewards[0]}')

        self.agent.save_model(model_file)


    def test(self, num_trials=1):
        ''' let agent act in the environment
            num_trials: may need multiple trials to get average
        '''
        print('Generate seeds at one time:', flush=True)
        all_rewards, all_seeds = self.play_game(num_trials, 0.0, False, time_usage = True, one_time = True)
        print(f'Number of trials: {num_trials}')
        print(f'Graph path: {", ".join(g.path_graph for g in self.env.graphs)}')
        cnt = 0
        for a_r, a_s in zip(all_rewards, all_seeds):
            print(f'Seeds: {a_s} | Reward: {a_r}')
            if len(self.env.graphs) > 1:
                cnt += 1
                if cnt == len(self.env.graphs):
                    print('')
                    cnt = 0

        print('Generate seed one by one:', flush=True)
        all_rewards, all_seeds = self.play_game(num_trials, 0.0, False, time_usage = True, one_time = False)
        print(f'Number of trials: {num_trials}')
        print(f'Graph path: {", ".join(g.path_graph for g in self.env.graphs)}')
        cnt = 0
        for a_r, a_s in zip(all_rewards, all_seeds):
            print(f'Seeds: {a_s} | Reward: {a_r}')
            if len(self.env.graphs) > 1:
                cnt += 1
                if cnt == len(self.env.graphs):
                    print('')
                    cnt = 0
