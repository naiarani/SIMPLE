import gym
import numpy as np
import itertools
from stable_baselines import logger
from .classes import *
import random

class CalicoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose=False, manual=False):
        super(CalicoEnv, self).__init__()
        self.name = 'calico'
        self.manual = manual

        self.tiles_per_player = 2
        self.tiles_types = 36
        self.board_size = (5, 5)

        self.n_rounds = 25

        self.n_players = 3  # player 0 and player 1
        self.current_player_num = 0

        self.colors = ['red', 'yellow', 'green', 'light blue', 'navy', 'purple']
        self.patterns = ['stripes', 'dots', 'fern', 'quatrefoil', 'flowers', 'vines']
        unique_tiles = list(itertools.product(self.colors, self.patterns))
        self.contents = [{'color': color, 'pattern': pattern} for color, pattern in unique_tiles for _ in range(3)]

        self.total_tiles = len(self.contents)

        self.quilt_size = 5
        self.grid_shape = (self.quilt_size, self.quilt_size)
        self.num_squares = self.quilt_size * self.quilt_size

        self.action_space = gym.spaces.Discrete(25)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.grid_shape[0], self.grid_shape[1], len(self.colors) * len(self.patterns)), dtype=np.float32)
        self.verbose = verbose

        self.quilt_boards = [np.zeros(self.board_size, dtype=int) for _ in range(self.n_players)]

        self.player_hands = [self.draw_starting_tiles(self.tiles_per_player) for _ in range(self.n_players)]

    @property
    def observation(self):
        player_observation = np.zeros((self.grid_shape[0], self.grid_shape[1], len(self.colors) * len(self.patterns)), dtype=int)

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                tile = self.quilt_boards[self.current_player_num][row, col]
                if tile != 0:
                    color_index = self.colors.index(tile['color'])
                    pattern_index = self.patterns.index(tile['pattern'])
                    player_observation[row, col, color_index * len(self.patterns) + pattern_index] = 1

        for idx, tile in enumerate(self.player_hands[self.current_player_num]):
            color_index = self.colors.index(tile['color'])
            pattern_index = self.patterns.index(tile['pattern'])
            player_observation[self.grid_shape[0] - 1, idx, color_index * len(self.patterns) + pattern_index] = 1

        return player_observation

    @property
    def legal_actions(self):
        legal_actions = np.zeros(self.board_size[0] * self.board_size[1], dtype=int)
        current_player_hand = self.player_hands[self.current_player_num]

        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.quilt_boards[self.current_player_num][row, col] == 0:
                    if current_player_hand:
                        legal_actions[row * self.board_size[1] + col] = 1

        return legal_actions

    def step(self, action):
        reward = 0
        done = False

        if self.legal_actions[action] == 0:
            # Penalize the current player if the action is illegal
            reward = -1
            done = True
        else:
            # Get the current player's hand and the tile to place
            current_player_hand = self.player_hands[self.current_player_num]
            tile_to_place = current_player_hand.pop(action)

            # Place the tile on the quilt board
            self.place_tile(tile_to_place)

            # Check if the game is over
            if self.is_game_over():
                done = True
                reward = self.calculate_final_reward()

            # Update current player number
            self.current_player_num = (self.current_player_num + 1) % self.n_players

        # Return observation, reward, done flag, and additional information
        return self.observation, reward, done, {}

    def reset(self):
        # Reset quilt boards to empty
        self.quilt_boards = [np.zeros(self.board_size, dtype=int) for _ in range(self.n_players)]
        self.player_hands = [self.draw_starting_tiles(self.tiles_per_player) for _ in range(self.n_players)]
        self.current_player_num = 0
        self.turns_taken = 0

        logger.debug(f'\n\n---- NEW GAME ----')

        # Return the initial observation
        return self.observation

    def draw_starting_tiles(self, n):
        random.shuffle(self.contents)
        drawn_tiles = self.contents[:n]
        self.contents = self.contents[n:]
        return drawn_tiles

    def draw_tile(self):
        if self.contents:
            tile = self.contents.pop()
            return {'color': self.colors.index(tile['color']), 'pattern': self.patterns.index(tile['pattern'])}
        else:
            return None

    def render(self):
        # Output quilt boards for each player
        for player_num, quilt_board in enumerate(self.quilt_boards):
            print(f"Player {player_num} Quilt Board:")
            for row in range(self.board_size[0]):
                for col in range(self.board_size[1]):
                    tile = quilt_board[row, col]
                    if tile != 0:
                        color_index = self.colors.index(tile['color'])
                        pattern_index = self.patterns.index(tile['pattern'])
                        print(f"{self.colors[color_index]} {self.patterns[pattern_index]}", end=" ")
                    else:
                        print("Empty", end=" ")
                print()  # Newline after each row
            print()  # Newline after each player's quilt board

        # Output hands for each player
        for player_num, hand in enumerate(self.player_hands):
            print(f"Player {player_num} Hand:")
            for tile in hand:
                color_index = self.colors.index(tile['color'])
                pattern_index = self.patterns.index(tile['pattern'])
                print(f"{self.colors[color_index]} {self.patterns[pattern_index]}", end=" ")
            print()  # Newline after each player's hand

    def check_winner(self):
        scores = [self.calculate_score(player_num) for player_num in range(self.n_players)]
        max

