#!/bin/env python3

import random

AGENT = 1
OPPONENT = -1
NO_PLAYER = 0


class Game:    
    def __init__(self, game_state=None):
        if game_state is None:
            game_state = [
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            ]
        self.state = game_state
    
    def __str__(self):
        return str(self.state)

    def is_draw(self):
        return len([field for field in self.state if field == NO_PLAYER]) == 0

    def is_finished(self):
        return self.get_winner() != NO_PLAYER or self.is_draw()

    def valid_moves(self):
        return [i for i in range(9) if self.state[i] == NO_PLAYER]

    def make_move(self, field, player):
        next = list(self.state)
        next[field] = player
        return Game(next)

    def get_winner(self):
        state = self.state
        for i in range(3):
            if state[i * 3] == state[i * 3 + 1] == state[i * 3 + 2] == state[i * 3] != NO_PLAYER:
                return state[i * 3]
            if state[i] == state[i + 3] == state[i + 6] == state[i] != NO_PLAYER:
                return state[i]
            if state[0] == state[4] == state[8] == state[0] != NO_PLAYER:
                return state[0]
            if state[2] == state[4] == state[6] == state[2] != NO_PLAYER:
                return state[2]

        return NO_PLAYER


def play_games(policy, opponent_policy, num_games=100):
    games_won = 0
    draw = 0
    # Play games
    for i in range(num_games):
        game = Game()
        # 50% chance opponent starts
        if random.random() > 0.5:
            game = game.make_move(opponent_policy(game), OPPONENT)

        while not game.is_finished():
            # First players turn
            game = game.make_move(policy(game), AGENT)
            if game.is_finished():
                break
            # Other players turn
            game = game.make_move(opponent_policy(game), OPPONENT)

        if game.get_winner() == 0:
            draw = draw + 1
        if game.get_winner() > 0:
            games_won = games_won + 1

    return games_won, draw


def reward(game):
    return max(game.get_winner(), 0)


def random_policy(game):
    return random.choice(game.valid_moves())


class ValuePolicy:
    DEFAULT_VALUE = 0.5

    def __init__(self):
        self.values = {}

    def policy(self, game):
        move_values = {}
        moves = game.valid_moves()
        for move in moves:
            next = game.make_move(move, AGENT)
            move_values[move] = self.get_state_value(next)

        return max(move_values, key=move_values.get)

    def get_state_value(self, state):
        if str(state) not in self.values:
            return self.DEFAULT_VALUE

        return self.values[str(state)]

    def set_state_value(self, state, value):
        self.values[str(state)] = value

    def learn(self, states):
        # Actually perform the learning
        def temporal_difference(current_state_value, next_state_value):
            learning_rate = 0.1
            return current_state_value + learning_rate * (next_state_value - current_state_value)

        last_state = states[-1:][0]
        last_value = reward(last_state)
        self.set_state_value(last_state, last_value)
        # Got through every state from end to start
        for state in reversed(states[:-1]):
            value = self.get_state_value(state)
            last_value = temporal_difference(value, last_value)
            self.set_state_value(state, last_value)


def train(policy, opponent_policy, training_games=1000):
    for i in range(training_games):
        game = Game()
        states = []

        # 50% chance opponent starts
        if random.random() > 0.5:
            game = game.make_move(opponent_policy(game), OPPONENT)

        while not game.is_finished():
            # Our agent makes a move
            # but occasionally we make a random choice
            if random.random() < 0.5:
                game = game.make_move(random_policy(game), AGENT)
            else:
                game = game.make_move(policy.policy(game), AGENT)
            states.append(game)

            if game.is_finished():
                break

            game = game.make_move(opponent_policy(game), OPPONENT)
            states.append(game)

        policy.learn(states)


def save_pickle(values):
    import pickle
    pickle.dump(values, open("values.pickle", "wb"))

if __name__ == "__main__":

    policy = ValuePolicy()

    train(policy, random_policy, training_games=10000)

    games_won, draw = play_games(policy.policy, random_policy, 1000)

    print("Games won: %s" % games_won)
    print("Draw: %s" % draw)

    # won.append(games_won)
    # draws.append(draw)
    #
    # from matplotlib import pyplot as plt
    #
    # plt.ylabel("% Games won")
    # plt.xlabel("Training games")
    # plt.plot(
    #     list(map(lambda x: x * 100, range(200))),
    #     list(map(lambda x: x / 10, won))
    # )
    # plt.show()
