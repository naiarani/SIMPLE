
import gym
import numpy as np
import random

import config

from stable_baselines import logger

from .classes import *

class CalicoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(CalicoEnv, self).__init__()
        self.name = 'calico'
        self.manual = manual

        # Defining players
        self.n_players = 2; # two-player for simplicity
        self.player_scores = [0, 0]  # Assuming 2 players for now

        self.player_hands = [self.draw_starting_tiles(2) for _ in range(self.n_players)]

        # Defining tiles
        self.colors = ['red', 'yellow', 'green', 'light blue', 'navy', 'purple']
        self.patterns = ['stripes', 'dots', 'fern', 'quatrefoil', 'flowers', 'vines']
        unique_tiles = list(itertools.product(self.colors, self.patterns))
        self.contents = [{'color': color, 'pattern': pattern} for color, pattern in unique_tiles for _ in range(3)]

        # Defining 5x5 square "quilt" grid
        self.quilt_size = 5
        self.num_squares = self.quilt_length * self.quilt_length # Square grid board for simplicity
        self.grid_shape = (self.grid_length, self.grid_length)
        self.action_space = gym.spaces.Discrete(self.num_squares)

    # Obervation / discritizing the board
    @property
    def observation(self):
        if self.players[self.current_player_num].token.number == 1:
            position = np.array([x.number for x in self.board]).reshape(self.grid_shape)
        else:
            position = np.array([-x.number for x in self.board]).reshape(self.grid_shape)

        la_grid = np.array(self.legal_actions).reshape(self.grid_shape)
        out = np.stack([position,la_grid], axis = -1)
        return out

    @property
    def legal_actions(self):
        legal_actions = []
        for action_num in range(len(self.board)):
            if self.board[action_num].number==0: #empty square
                legal_actions.append(1)
            else:
                legal_actions.append(0)
        return np.array(legal_actions)
        

    ####### Define scoring based of simplified version including 4 ways to score #######
    def check_score(self):
        score = 0
        score += self.check_button_score()
        score += self.check_callie_score()
        score += self.check_rumi_score()
        score += self.check_coconut_score()
        return score

    def check_button_score(self):
        score = 0
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                # Check if the current tile has two neighbors of the same color
                if self.has_same_color_neighbors(row, col):
                    # Increment score by 3 for each set of 3 same-color adjacent tiles
                    score += 3
        return score

    def has_same_color_neighbors(self, row, col):
        # Check if the current tile has two neighbors of the same color
        color = self.board[row][col]['color']
        adjacent_tiles = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        same_color_neighbors = 0
        for r, c in adjacent_tiles:
            if 0 <= r < len(self.board) and 0 <= c < len(self.board[0]):
                if self.board[r][c]['color'] == color:
                    same_color_neighbors += 1
        return same_color_neighbors >= 2

    def check_callie_score(self):
        score = 0
        # Check for "callie" scoring
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.has_callie_shape(row, col):
                    score += 3
        return score

    def has_callie_shape(self, row, col):
        # Check if the tile at the given position forms an L shape with the same pattern
        pattern = self.board[row][col]['pattern']
        if pattern not in ['stripes', 'dots']:
            return False
        
        # Define the offsets for the tiles in the L shape
        offsets = [(0, 1), (1, 0), (1, 1)]
        
        # Check if the adjacent tiles have the same pattern
        for r_off, c_off in offsets:
            r, c = row + r_off, col + c_off
            if not (0 <= r < len(self.board) and 0 <= c < len(self.board[0]) and self.board[r][c]['pattern'] == pattern):
                return False
        
        return True

    def check_rumi_score(self):
        score = 0
        # Check for "rumi" scoring
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.has_rumi_shape(row, col):
                    score += 5
        return score

    def has_rumi_shape(self, row, col):
        # Check if the tile at the given position forms a straight line of the same pattern
        pattern = self.board[row][col]['pattern']
        if pattern not in ['fern', 'quatrefoil']:
            return False
        
        # Define the directions to check for the straight line
        directions = [(0, 1), (1, 0)]
        
        # Check if the adjacent tiles have the same pattern in any of the directions
        for r_dir, c_dir in directions:
            curr_row, curr_col = row, col
            count = 1  # Count the number of tiles with the same pattern
            while True:
                curr_row += r_dir
                curr_col += c_dir
                if not (0 <= curr_row < len(self.board) and 0 <= curr_col < len(self.board[0])):
                    break
                if self.board[curr_row][curr_col]['pattern'] == pattern:
                    count += 1
                else:
                    break
            if count >= 3:
                return True
        
        return False

    def check_coconut_score(self):
        score = 0
        # Check for "coconut" scoring
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.has_coconut_shape(row, col):
                    score += 7
        return score

    def has_coconut_shape(self, row, col):
        # Check if the tile at the given position forms a cluster of the same pattern with 5 or more adjacent tiles
        pattern = self.board[row][col]['pattern']
        if pattern not in ['flowers', 'vines']:
            return False
        
        # Define the directions to check for adjacent tiles
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Depth-first search to find all adjacent tiles with the same pattern
        def dfs(r, c, visited):
            visited.add((r, c))
            count = 1  # Count the number of adjacent tiles with the same pattern
            for r_dir, c_dir in directions:
                new_r, new_c = r + r_dir, c + c_dir
                if (0 <= new_r < len(self.board) and 0 <= new_c < len(self.board[0])
                        and (new_r, new_c) not in visited
                        and self.board[new_r][new_c]['pattern'] == pattern):
                    count += dfs(new_r, new_c, visited)
            return count
        
        visited = set()
        if dfs(row, col, visited) >= 5:
            return True
        
        return False


        # Initialize game state
        self.reset()

    def reset(self):
        # Reset quilt board and player scores
        self.quilt_board = np.zeros(self.quilt_size, dtype=int)
        self.player_scores = [0, 0]  # Reset scores for 2 players
        self.player_hands = [self.draw_starting_tiles(2) for _ in range(self.n_players)]
        return self.quilt_board

    def draw_starting_tiles(self, n):
        return [self.draw_tile() for _ in range(n)]
    
    def draw_tile(self):
        if self.contents:
            return self.contents.pop()
        else:
            return None


    def step(self, action):
        color_index = action // len(self.patterns)
        pattern_index = action % len(self.patterns)
        color = self.colors[color_index]
        pattern = self.patterns[pattern_index]
        
        # Find an empty spot on the quilt board to place the tile
        empty_spots = np.argwhere(self.quilt_board == 0)
        if len(empty_spots) == 0:
            # No empty spots left, terminate the episode
            return self.quilt_board, 0, True, {}
        
        # Choose a random empty spot to place the tile
        row, col = empty_spots[np.random.randint(len(empty_spots))]
        
        # Place the chosen tile on the quilt board
        self.quilt_board[row, col] = action
        
        # Replenish players' hands
        for player_id in range(self.n_players):
            self.replenish_hand(player_id)
        
        # Calculate the score for the current state
        score = self.check_score()
        
        # Determine if the episode is done (if the quilt board is full)
        done = np.count_nonzero(self.quilt_board) == self.num_squares
        
        return self.quilt_board, score, done, {}


    def render(self, mode='human'):
        # Display the current state of the quilt board
        print("Current Quilt Board:")
        print(self.quilt_board)

    def close(self):
        pass
