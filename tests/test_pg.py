from margam.pg import PolicyPlayer, PolicyGradientTrainer
from rl import build_game_handler        

def test_train_pg():
    gh = build_game_handler("tic_tac_toe")
    agent = PolicyPlayer(gh, name="dqn-agent", model=None)
    trainer = PolicyGradientTrainer(
        game_type = gh.game_type.value,
        hyperparameters = ...,  # fill with hps for short training
        agent = agent
        opponents = [RandomPlayer(gh)]
        save_to_disk=False,
    )

    print(f"Training agent with {trainer.type} to play {game_type}")
    trainer.train()