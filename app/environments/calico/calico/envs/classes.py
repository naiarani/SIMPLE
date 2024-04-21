# Calico classes

import random

class CalicoTiles:
    def __init__(self):
        self.tiles = []

    def create_tiles(self):
        colors = ['red', 'yellow', 'green', 'light blue', 'navy', 'purple']
        patterns = ['stripes', 'dots', 'fern', 'quatrefoil', 'flowers', 'vines']
        for color in colors:
            for pattern in patterns:
                for _ in range(3):  # Create 3 tiles for each color-pattern combination
                    self.tiles.append({'color': color, 'pattern': pattern})

    def shuffle(self):
        random.shuffle(self.tiles)

    def draw(self, n):
        drawn = []
        for _ in range(n):
            if self.tiles:
                drawn.append(self.tiles.pop())
        return drawn

    def size(self):
        return len(self.tiles)


class Player:
    def __init__(self, id):
        self.id = id
        self.score = 0
        self.hand = Hand()
        self.position = Position()


class Hand:
    def __init__(self):
        self.cards = []

    def add(self, cards):
        self.cards.extend(cards)

    def size(self):
        return len(self.cards)

    def pick(self, name):
        for i, card in enumerate(self.cards):
            if card.name == name:
                return self.cards.pop(i)


class Position:
    def __init__(self):
        self.cards = []

    def add(self, cards):
        self.cards.extend(cards)

    def size(self):
        return len(self.cards)

    def pick(self, name):
        for i, card in enumerate(self.cards):
            if card.name == name:
                return self.cards.pop(i)


class Card:
    def __init__(self, id, order, name):
        self.id = id
        self.order = order
        self.name = name


class CalicoTile(Card):
    def __init__(self, id, order, name, color, pattern):
        super().__init__(id, order, name)
        self.color = color
        self.pattern = pattern

    @property
    def symbol(self):
        return f"{self.color[0].upper()}{self.pattern[0].upper()}"
