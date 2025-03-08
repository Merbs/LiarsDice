from margam.rl import build_game_handler
from margam.player import RandomPlayer

gh = build_game_handler("tic_tac_toe")
state = gh.game.new_initial_state()

def test_something():
    p = RandomPlayer(gh)
    p.get_move(state)
