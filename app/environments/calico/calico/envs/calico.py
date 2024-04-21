
import gym
import numpy as np
import config
import itertools

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
        self.current_player_num = 0
        self.cards_per_player = 2
        self.card_types = 36

        self.n_rounds = 25
        self.max_score = 100

        # Defining tiles
        self.colors = ['red', 'yellow', 'green', 'light blue', 'navy', 'purple']
        self.patterns = ['stripes', 'dots', 'fern', 'quatrefoil', 'flowers', 'vines']
        unique_tiles = list(itertools.product(self.colors, self.patterns))
        self.contents = [{'color': color, 'pattern': pattern} for color, pattern in unique_tiles for _ in range(3)]

        # Defining 5x5 square "quilt" grid
        self.quilt_size = 5
        self.num_squares = self.quilt_size * self.quilt_size # Square grid board for simplicity
        self.grid_shape = (self.quilt_size, self.quilt_size)
        
        self.player_hands = [self.draw_starting_tiles(2) for _ in range(self.n_players)]

        self.action_space = gym.spaces.Discrete(self.num_squares)
        self.observation_space = gym.spaces.Box(0, 1, (self.num_squares * self.card_types + self.n_players + self.action_space.n,))
        self.verbose = verbose


    # Obervation / discritizing the board
    @property
    def observation(self):
        # Initialize the observation array
        obs = np.zeros((self.num_squares, self.card_types), dtype=int)
        player_num = self.current_player_num
        
        # Add the tiles on the quilt board to the observation
        for row in range(len(self.quilt_board)):
            for col in range(len(self.quilt_board[row])):
                tile = self.quilt_board[row, col]
                if tile != 0:  # Tile is present
                    color_index = tile // len(self.patterns)
                    pattern_index = tile % len(self.patterns)
                    obs[row * len(self.quilt_board) + col][color_index * len(self.patterns) + pattern_index] = 1
        
        # Flatten the observation array
        ret = obs.flatten()
        
        # Return the flattened observation array
        return ret
    
    @property
    def legal_actions(self):
        # Initialize the legal actions array
        legal_actions = np.zeros(self.num_squares)
        player_num = self.current_player_num

        
        # Check each position on the quilt board
        for row in range(len(self.quilt_board)):
            for col in range(len(self.quilt_board[row])):
                # Check if the position is empty
                if self.quilt_board[row, col] == 0:
                    # Set the corresponding index in the legal actions array to 1
                    legal_actions[row * len(self.quilt_board) + col] = 1
        
        # Return the legal actions array
        return legal_actions

        

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

    def draw_starting_tiles(self, n):
        return [self.draw_tile() for _ in range(n)]

    def reset(self):
        self.round = 0
        self.deck = Deck(self.contents)
        self.discard = Discard()
        self.players = []
        self.action_bank = []
        self.player_hands = [self.draw_starting_tiles(2) for _ in range(self.n_players)]

    
        player_id = 1
        for p in range(self.n_players):
            self.players.append(Player(str(player_id)))
            player_id += 1
    
        self.current_player_num = 0
        self.done = False
        self.reset_round()
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation



    def step(self, action):
      reward = [0] * self.n_players
      done = False

      # Check move legality
      if self.legal_actions[action] == 0:
          # Penalize the current player and reward others if the action is illegal
          reward = [1.0 / (self.n_players - 1)] * self.n_players
          reward[self.current_player_num] = -1
          done = True
      else:
          # Play the card(s)
          self.action_bank.append(action)

          if len(self.action_bank) == self.n_players:
              logger.debug(f'\nThe chosen cards are now played simultaneously')
              for i, action in enumerate(self.action_bank):
                  # Assuming you have a method to convert action to card(s)
                  player = self.players[i]
                  pickup_chopsticks, first_card, second_card = self.convert_action(action)
                  self.play_card(first_card, player)

                  if pickup_chopsticks:
                      self.pickup_chopsticks(player)
                      self.play_card(second_card, player)

              self.action_bank = []
              self.switch_hands()
          
          # Update current player number
          self.current_player_num = (self.current_player_num + 1) % self.n_players

          # Update turns taken
          if self.current_player_num == 0:
              self.turns_taken += 1

          # Check if the round is over
          if self.turns_taken == self.cards_per_player:
              # Score the round
              self.score_round()

              # Check if the game is over
              if self.round >= self.n_rounds:
                  # Score puddings and end the game
                  self.score_puddings()
                  reward = self.score_game()
                  done = True
              else:
                  # Reset the round
                  self.render()
                  self.reset_round()

      self.done = done

      # Return observation, reward, done flag, and additional information
      return self.observation, reward, done, {}


    def render(self, mode='human', close=False):
        if close:
            return
    
        if self.turns_taken < self.cards_per_player:
            logger.debug(f'\n\n-------ROUND {self.round} : TURN {self.turns_taken + 1}-----------')
            logger.debug(f"It is Player {self.current_player.id}'s turn to choose")
        else:
            logger.debug(f'\n\n-------FINAL ROUND {self.round} POSITION-----------')
    
        for p in self.players:
            logger.debug(f'\nPlayer {p.id}\'s hand')
            if p.hand.size() > 0:
                logger.debug('  '.join([str(card.order) + ': ' + card.symbol for card in sorted(p.hand.cards, key=lambda x: x.id)]))
            else:
                logger.debug('Empty')
    
            logger.debug(f'Player {p.id}\'s position')
            if p.position.size() > 0:
                logger.debug('  '.join([str(card.order) + ': ' + card.symbol + ': ' + str(card.id) for card in sorted(p.position.cards, key=lambda x: x.id)]))
            else:
                logger.debug('Empty')
    
        logger.debug(f'\n{self.deck.size()} cards left in deck')
        logger.debug(f'{self.discard.size()} cards discarded')
    
        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')
    
        if self.done:
            logger.debug(f'\n\nGAME OVER')
    
        if self.turns_taken == self.cards_per_player:
            for p in self.players:
                logger.debug(f'Player {p.id} points: {p.score}')


    def close(self):
        pass
