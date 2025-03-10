import yaml

from margam.pg import PolicyGradientTrainer
from margam.rl import build_game_handler
from margam.player import RandomPlayer

def get_short_training_hp():
    with open("input_files/tic-tac-toe-pg.yaml", "r") as f:
        hp = yaml.safe_load(f)
    return hp

def test_train_pg():
    trainer = PolicyGradientTrainer(
        hyperparameters = get_short_training_hp(),
        save_to_disk=False,
    )
    trainer.MAX_EPISODES = 100
    trainer.train()
    assert trainer.step > 0
