from margam.rl import build_game_handler
from margam.player import RandomPlayer, create_player

import pytest

gh = build_game_handler("liars_dice")
state = gh.game.new_initial_state()

@pytest.mark.parametrize(
    "player_type", [
        "conservative", "random", "minimax", "spammer"
    ]
)
def test_random_player(player_type):
    p = create_player(player_type, gh)
    p.get_move(state)
