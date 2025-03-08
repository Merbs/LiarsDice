from margam.pg import PolicyPlayer, PolicyGradientTrainer
from rl import build_game_handler        

def get_short_training_hp():
    with open("input_files/tic-tac-toe-pg.yaml", "r") as f:
        hp = yaml.safe_load(f)

def test_train_pg():
    trainer = PolicyGradientTrainer(
        game_handler = build_game_handler("tic_tac_toe"),
        hyperparameters = get_short_training_hp(),
        agent = PolicyPlayer(gh, name="dqn-agent", model=None),
        opponents = [RandomPlayer(gh)],
        save_to_disk=False,
    )
    trainer.MAX_EPISODES = 1000
    trainer.train()
    assert trainer.step > 0
