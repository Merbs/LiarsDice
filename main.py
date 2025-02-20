import numpy as np
import random
import yaml
import click
import pyspiel

from margam.player import HumanPlayer, MiniMax
from margam.rl import build_game_handler

@click.group()
def main():
    pass


@main.command()
@click.option(
    "-g",
    "--game-type",
    type=click.Choice(list(pyspiel.registered_names()), case_sensitive=False),
    default="connect_four",
    show_default=True,
    help="game type",
)
@click.option(
    "-o",
    "--opponent",
    type=click.Choice(["human", "minimax", "dqn", "pg"], case_sensitive=False),
    default="minimax",
    show_default=True,
    help="opponent type",
)
@click.option(
    "-d", "--depth", type=int, default=2, show_default=True, help="Depth for minimax"
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Model file to load for AI player",
)
@click.option("--second", is_flag=True, default=False, help="Play as second player")
def play(game_type, opponent, depth, model, second):

    gh = build_game_handler(game_type)

    # Intialize players
    human = HumanPlayer(gh, name="Marcel")

    opponent = opponent.lower()
    if opponent == "minimax":
        opponent = MiniMax(gh, name="Maximus", max_depth=depth)
    elif opponent == "human":
        opponent = HumanPlayer(gh, "Opponent")
    elif opponent == "pg":
        from margam.train_pg import PolicyPlayer
        from keras.models import load_model

        opponent = PolicyPlayer(name="PG")
        if game_type == "liars_dice":
            from margam.train_pg import initialize_model
            opponent.model = initialize_model(
                game_type, {"ACTOR_CRITIC":False}, show_model=True)
        else:
            opponent.model = load_model(model)
        opponent.model.summary()
    elif opponent == "dqn":
        from margam.train_dqn import DQNPlayer
        from keras.models import load_model

        opponent = DQNPlayer(name="DQN")
        opponent.model = load_model(model)
        opponent.model.summary()

    
    tsns = gh.generate_episode_transitions(human, opponent, 1 if second else 0)

    winner = np.argmax(state.returns())
    print(f"{players[winner].name} won!")


@main.command()
@click.option(
    "-g",
    "--game-type",
    type=click.Choice(list(pyspiel.registered_names()), case_sensitive=False),
    default="tic_tac_toe",
    show_default=True,
    help="game type",
)
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(["dqn", "pg"], case_sensitive=False),
    default="dqn",
    show_default=True,
    help="Reinforcement learning algorithm",
)
@click.option(
    "-h",
    "--hyperparameter-file",
    type=str,
    default="hyperparams.yaml",
    show_default=True,
    help="YAML file with hyperparameter values",
)
def train(game_type, algorithm, hyperparameter_file):

    with open(hyperparameter_file, "r") as f:
        hp = yaml.safe_load(f)

    if algorithm == "dqn":
        from margam.train_dqn import train_dqn
        train_dqn(game_type, hp[game_type])
    elif algorithm == "pg":
        from margam.train_pg import train_pg
        train_pg(game_type, hp[game_type])


if __name__ == "__main__":
    main()
