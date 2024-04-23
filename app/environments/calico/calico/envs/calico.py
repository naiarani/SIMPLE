
import gym
import numpy as np
import config
import itertools
from stable_baselines import logger
from .classes import *
import random

class CalicoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose=False, manual=False):
        super(CalicoEnv, self).__init__()
        self.board_size = (5, 5)
        self.num_colors = 6
        self.num_patterns = 6
        self.num_tiles = 36
        self.hand_size = 2
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.board_size[0], self.board_size[1], self.num_colors + self.num_patterns), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.board_size[0] * self.board_size[1])
        self.reset()
        self.total_scores = {player_num: 0 for player_num in range(self.n_players)}


    def reset(self):
        self.board = np.zeros(self.board_size, dtype=int)
        self.available_spots = np.ones(self.board_size, dtype=int)
        self.current_player = 0
        self.tiles_bank = self._generate_tiles_bank()
        self.player_hands = [self._draw_tiles() for _ in range(2)]
        return self._get_observation()

    def step(self, action):
        row = action // self.board_size[1]
        col = action % self.board_size[1]
        if self.available_spots[row, col] == 0:
            return self._get_observation(), 0, False, {}
        else:
            # Check if the tile is in the current player's hand
            if self.board[row, col] not in [tile['id'] for tile in self.player_hands[self.current_player]]:
                return self._get_observation(), 0, False, {}
            else:
                # Place the tile on the board
                self.board[row, col] = 1
                # Mark the spot as occupied
                self.available_spots[row, col] = 0
                # Check for point scoring
                color_score = self._check_color_groups(row, col)
                pattern_score = self._check_straight_lines(row, col)
                total_score = color_score + pattern_score
                # Remove the placed tile from the player's hand
                self.player_hands[self.current_player] = [tile for tile in self.player_hands[self.current_player] if tile['id'] != self.board[row, col]]
                # Draw a new tile for the player
                new_tile = self._draw_tile()
                self.player_hands[self.current_player].append(new_tile)
                # Switch to the next player
                self.current_player = (self.current_player + 1) % 2
                # Check if the game is over
                game_over = np.sum(self.board) == self.board_size[0] * self.board_size[1]
                # Return observation, reward, done, info
                return self._get_observation(), total_score, game_over, {}
        reward, done = self.check_score()
        self.total_scores[self.current_player_num] += reward

    def _get_observation(self):
        observation = np.zeros((self.board_size[0], self.board_size[1], self.num_colors + self.num_patterns), dtype=np.float32)
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.board[row, col] == 0:
                    # Spot is empty
                    observation[row, col, :] = 1
                else:
                    # Spot is occupied
                    observation[row, col, -1] = 1
        return observation

    def _generate_tiles_bank(self):
        tiles_bank = [{'id': i % self.num_tiles + 1, 'color': i % self.num_colors + 1, 'pattern': i % self.num_patterns + 1} for i in range(self.num_tiles)]
        random.shuffle(tiles_bank)
        return tiles_bank

    def _draw_tiles(self):
        return [self._draw_tile() for _ in range(self.hand_size)]

    def _draw_tile(self):
        return self.tiles_bank.pop()

    def _check_color_groups(self, row, col):
        color_id = self.board[row, col]
        score = 0
        # Check horizontally
        if col >= 2 and self.board[row, col-1] == color_id and self.board[row, col-2] == color_id:
            score += 3
        if col <= self.board_size[1] - 3 and self.board[row, col+1] == color_id and self.board[row, col+2] == color_id:
            score += 3
        # Check vertically
        if row >= 2 and self.board[row-1, col] == color_id and self.board[row-2, col] == color_id:
            score += 3
        if row <= self.board_size[0] - 3 and self.board[row+1, col] == color_id and self.board[row+2, col] == color_id:
            score += 3
        return score

    def _check_straight_lines(self, row, col):
        pattern_id = self.board[row, col]
        score = 0
        # Check horizontally
        count = 1
        for c in range(col - 1, -1, -1):
            if self.board[row, c] == pattern_id:
                count += 1
            else:
                break
        for c in range(col + 1, self.board_size[1]):
            if self.board[row, c] == pattern_id:
                count += 1
            else:
                break
        if count >= 4:
            score += 5
        # Check vertically
        count = 1
        for r in range(row - 1, -1, -1):
            if self.board[r, col] == pattern_id:
                count += 1
            else:
                break
        for r in range(row + 1, self.board_size[0]):
            if self.board[r, col] == pattern_id:
                count += 1
            else:
                break
        if count >= 4:
            score += 5
        return score

    def check_score(self):
        score = 0
        score += self.check_color_groups()
        score += self.check_pattern_sequences()
        done = self.check_game_over()
        return score, done

    def check_winner(self):
        # Compare total scores of all players to determine the winner
        max_score = max(self.total_scores.values())
        winners = [player_num for player_num, score in self.total_scores.items() if score == max_score]
        if len(winners) == 1:
            return winners[0]
        else:
            return None



