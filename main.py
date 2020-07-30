from player import RLAgent, Human
from game import Game
import random
import numpy as np
from time import time
import json
import sys
import os

game_board_width = 2
game_board_height = 2
gamma = 0.9
learning_rate = 0.0001
batch_size = 150
copy_weights_switch = 1000
replay_memory_size = 10000

def save_data(data, file_name):
    if(not os.path.exists("./_rewards")):
        os.makedirs("_rewards")
    with open("_rewards/"+file_name, "w") as f:
        json.dump(data, f)

'''
This is an attempt to train a game of size 2x2 board. Input State size: 16 = 12 edges + 4 boxes.
Output Action space: Gives Q-values for each edge on the board. Here action-space = 12
'''

def train(number_of_episodes = 100000):
    random.seed(0)
    p1 = RLAgent("RLAgent1",gamma, learning_rate, batch_size, copy_weights_switch, 16, [120,200,120], 12, replay_memory_size)
    p2 = RLAgent("RLAgent2",gamma, learning_rate, batch_size, copy_weights_switch, 16, [120,200,120], 12, replay_memory_size)

    number_of_episodes = int(number_of_episodes)
    win_count = {p1.name:0, p2.name:0}
    game_count = 0
    train_start_time = time()
    epsilon = 1
    epsilon_min = 0.01
    number_of_moves_list = []
    total_rewards = {p1.name:[], p2.name:[]}
    for episode_index in range(1,number_of_episodes+1):
        game_count+=1
        game = Game(game_board_height, game_board_width, p1, p2) if episode_index%2 == 0 else Game(game_board_height, game_board_width, p2, p1)
        active_chache = {p1.name:None, p2.name:None}
        number_of_moves = 0
        episode_rewards = {p1.name:0, p2.name:0}
        while not game.get_status()["game_over"]:
            number_of_moves+=1
            state = game.get_state()
            if(active_chache[game.active_player.name] is not None):
                active_chache[game.active_player.name]['next_state'] = state
                active_chache[game.active_player.name]['done'] = False
                game.active_player.add_to_memory(active_chache[game.active_player.name])

            if episode_index <= number_of_episodes // 7:
                max_eps = 1
            elif episode_index <= number_of_episodes // 4:
                max_eps = 0.6
            elif episode_index <= number_of_episodes // 2:
                max_eps = 0.1
            else:
                max_eps = 0.05
            epsilon = round(max(max_eps - round(episode_index*(max_eps-epsilon_min)/number_of_episodes, 3), epsilon_min), 3)

            epsilon = max(epsilon_min, epsilon)

            selected_edge = int(game.active_player.get_edge(state, epsilon = epsilon))
            status = game.play(selected_edge)
            game_over = status['game_over']
            if(status['invalid_move']):
                r = game.invalid_move_reward
            elif game_over:
                if status['winner'] == 0:
                    r = game.tie_reward
                elif status['winner'] == game.active_player.id:
                    r = game.winning_reward
                    win_count[game.active_player.name]+=1
                else:
                    r = game.losing_reward
            else:
                if status['double_closed']:
                    r = game.double_closed_reward
                elif status['box_closed']:
                    r = game.box_closed_reward
                else:
                    r = 0

            active_chache[game.active_player.name]= {"state": state, "action": selected_edge, "reward": r}
            episode_rewards[game.active_player.name]+=r
            
            # learn
            game.active_player.learn(batch_size)

            if not game_over:
                game.switch_player()

        active_chache[game.active_player.name]['next_state'] = game.init_state
        active_chache[game.active_player.name]['done'] = True
        game.active_player.add_to_memory(active_chache[game.active_player.name])

        active_chache[game.inactive_player.name]['next_state'] = game.init_state
        active_chache[game.inactive_player.name]['done'] = True
        active_chache[game.inactive_player.name]['reward'] = -r
        game.inactive_player.add_to_memory(active_chache[game.inactive_player.name])
        episode_rewards[game.inactive_player.name]-=r


        total_rewards[p1.name].append(episode_rewards[p1.name])
        total_rewards[p2.name].append(episode_rewards[p2.name])
        number_of_moves_list.append(number_of_moves)
        if(episode_index%100 == 0):
            print("\nTotal games: ", game_count)
            print("Average number of moves last 10 games: ", np.average(number_of_moves_list[-10:]), " Min: ", min(number_of_moves_list[-10:]), " Max: ", max(number_of_moves_list[-10:]))
            print("win_stats", win_count)
            print("Current epsilon value: ", epsilon)
            print("Average reward of last 10 games for player 1: ", np.average(total_rewards[p1.name][-10:]))
            print("Average reward of last 10 games for player 2: ", np.average(total_rewards[p2.name][-10:]))
            save_data(total_rewards[p1.name], p1.name+".txt")
            save_data(total_rewards[p2.name], p2.name+".txt")

            training_time = time() - train_start_time
            minutes = int(training_time // 60)
            seconds = int(training_time % 60)
            if seconds < 10:
                seconds = '0{}'.format(seconds)
            print("Time elapsed: {m}:{s} minutes".format(m=minutes, s=seconds))

        if(episode_index%(number_of_episodes/20) == 0):
            print("\n*****Backing up temporary models...*****")
            p1.save_models()
            p2.save_models()

    print("Training done")
    print("Saving models")

    p1.save_models()
    p2.save_models()

    training_time = time() - train_start_time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    if seconds < 10:
        seconds = '0{}'.format(seconds)
    print('Training took {m}:{s} minutes'.format(m=minutes, s=seconds))


def play(play_with = 0):
    random.seed(0)

    p1 = RLAgent("RLAgent2", gamma, learning_rate, batch_size, copy_weights_switch, 16, [100,180,100], 12, replay_memory_size)
    p1.load_models()

    p2 = None
    if(play_with == 1):
        p2 = RLAgent("RLAgent2", gamma, learning_rate, batch_size, copy_weights_switch, 16, [100,180,100], 12, replay_memory_size)
        p2.load_models()
    else:
        name = input("\nPlease enter your name: ")
        p2 = Human(name)
    num_of_games = 5
    print("Your player ID: ", p2.id)
    win_count = {p1.name:0, p2.name:0}
    total_rewards = {p1.name:[], p2.name:[]}
    move_count = {}
    for i in range(1, num_of_games+1):
        number_of_moves = 0
        game = Game(game_board_height, game_board_width, p1, p2)
        episode_rewards = {p1.name:0, p2.name:0}
        print("New Game!")
        while not game.get_status()["game_over"]:
            number_of_moves+=1
            state = game.get_state()
            game.draw_game()
            print("\nActive Player: ", game.active_player.name)
            selected_edge = int(game.active_player.get_edge(state, 0))
            print("Seleted Edge: ", selected_edge, )
            status = game.play(selected_edge)
            print(status)
            game_over = status['game_over']
            if(status['invalid_move']):
                r = game.invalid_move_reward
            elif game_over:
                if status['winner'] == 0:
                    r = game.tie_reward
                elif status['winner'] == game.active_player.id:
                    r = game.winning_reward
                    win_count[game.active_player.name]+=1
            else:
                if status['double_closed']:
                    r = game.double_closed_reward
                    print("Bingo! Double closed!")
                elif status['box_closed']:
                    r = game.box_closed_reward
                    print("Box closed!")
                else:
                    r = 0

            episode_rewards[game.active_player.name]+=r

            if not game_over:
                game.switch_player()
        total_rewards[p1.name].append(episode_rewards[p1.name])
        total_rewards[p2.name].append(episode_rewards[p2.name])
        move_count[i] = number_of_moves
        print("Game Over!! Winner: ", game.get_player_with_id(game.winner_id).name)
        print("Game board: ", game.get_state())
    print("Total games: ", num_of_games)
    print("Win_status: ", win_count)
    print("Move count: ", move_count)


if __name__ == "__main__":
    if(len(sys.argv)>1):
        option = sys.argv[1]
        if(option in ["train", "t", "0"]):
            train()
        else:
            play()
    else:
        train_net = input("Do you want to train the network? (y/n)").lower() in ["y", "yes"]
        if train_net:
            train()
        else:
            play()