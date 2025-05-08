import sys
import importlib

from dataclasses import dataclass
from enum import IntEnum
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from referee.game.board import Board, CellState
from referee.game.constants import BOARD_N


def load_agent(agent_name, color):
    mod = importlib.import_module(f"{agent_name}.program")
    return mod.Agent(color)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m match_game agent1 agent2")
        sys.exit(1)

    agent1_name, agent2_name = sys.argv[1], sys.argv[2]

    agent1 = load_agent(agent1_name, PlayerColor.RED)
    agent2 = load_agent(agent2_name, PlayerColor.BLUE)

    for turn in range(150):
        if turn % 2 == 0:
            move = agent1.action()
        else:
            move = agent2.action()
        
        agent1.update(PlayerColor.RED, move)
        agent2.update(PlayerColor.RED, move)
    
        agent1.print_board()