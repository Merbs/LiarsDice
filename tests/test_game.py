import pytest

from margam.rl import GameType, build_game_handler
from margam.player import RandomPlayer

@pytest.mark.parametrize("game_type", [gt.value for gt in GameType])
def test_random_players(game_type):
    """
    All our game types can be played with random players
    """
    gh = build_game_handler(game_type)
    players = [RandomPlayer(gh),RandomPlayer(gh)]
    ets = gh.generate_episode_transitions(players)
    assert(len(ets)>0)
