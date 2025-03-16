import numpy as np
import random
import yaml
import click
import pyspiel

from margam.player import HumanPlayer, MiniMax
from margam.rl import build_game_handler, GameType

@click.group()
def main():
    pass

@main.command()
@click.option(
    "-g",
    "--game-type",
    type=click.Choice([gt.value for gt in GameType], case_sensitive=False),
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
        opponent = PolicyPlayer(gh, name="PG", model=model)
        opponent.model.summary()
    elif opponent == "dqn":
        from margam.train_dqn import DQNPlayer
        opponent = DQNPlayer(gh, name="DQN", model=model)
        opponent.model.summary()

    players = [human, opponent]
    if second:
        players = list(reversed(players))
    tsns = gh.generate_episode_transitions(players)

    total_rewards = [sum(tsn.reward for tsn in tsn_list) for tsn_list in tsns]
    winner = np.argmax(total_rewards)
    print(f"{players[winner].name} won!")


@main.command()
@click.argument('hyperparameter-file')
def train(hyperparameter_file):

    with open(hyperparameter_file, "r") as f:
        hp = yaml.safe_load(f)

    try:
        game_type = hp["GAME"]
        algorithm = hp["ALGORITHM"]
        opponent_list = hp["OPPONENTS"]
    except KeyError as e:
        print(f"Hyperparameter file is missing field: {e}")
        sys.exit(1)

    gh = build_game_handler(game_type)
    opponents = [create_player(gh) for opp in opponent_list]

    if algorithm.lower() == "dqn":
        from margam.dqn import DQNPlayer, DQNTrainer
        agent = PolicyPlayer(gh, name="pg-agent", model=None)
        opponents = get_opponents(gh.game_type)
        trainer = DQNTrainer(
            game_type = gh,
            hyperparameters = hp,
            agent = agent,
            opponents = opponents,
            save_to_disk = True,
            )
    elif algorithm.lower() == "pg":
        from margam.pg import PolicyPlayer, PolicyGradientTrainer
        agent = PolicyPlayer(gh, name="dqn-agent", model=None)
        trainer = PolicyGradientTrainer(agent=agent)
    else:
        print(f"{algorithm} is not a supported algorithm. Options are dqn or pg.")
        sys.exit(1)

    print(f"Training agent with {trainer.type} to play {game_type}")
    trainer.train()



if __name__ == "__main__":
    main()
