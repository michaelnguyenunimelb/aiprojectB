# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from typing import Literal
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from referee.game.board import Board, CellState
from referee.game.constants import BOARD_N

import random

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
        
        self.board: dict[Coord, CellState] = dict() 

        for r in [0,BOARD_N-1]:
            for c in [0,BOARD_N-1]:
                self.board[Coord(r,c)] = CellState("LilyPad")
        
        for c in range(1,BOARD_N-1):
            self.board[Coord(0,c)] = CellState(PlayerColor.RED)
            self.board[Coord(BOARD_N-1,c)] = CellState(PlayerColor.BLUE)
            for r in [1,BOARD_N-2]:
                self.board[Coord(r,c)] = CellState("LilyPad")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")

            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a GROW action")
            
            # return MoveAction(
                #     Coord(0, 3),
                #     [Direction.Down]
                # )

        moves = generate_moves(self.board,self._color)

        bro = random.choice(moves)
        if isinstance(bro, GrowAction):                                
            return GrowAction()
        else:
            move, dest = bro
            start = move.coord

            return move



    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.
        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")
        
        match action:
            case MoveAction(coord, dirs):
                curr_coord = coord
                for di in dirs:
                    
                    if isinstance(self.board[curr_coord+di].state, PlayerColor):
                        curr_coord += di * 2
                    elif self.board[curr_coord+di].state == 'LilyPad':
                        curr_coord += di
                    else:
                        raise Exception("dawg")
                    
                
                self.board[coord] = CellState(None)
                self.board[curr_coord] = CellState(color)
                    
            case GrowAction():
                new_board = self.board.copy()
                for coord in self.board:
                    if self.board[coord].state == color:
                        for direc in Direction:
                            try: 
                                new = coord + direc
                                if new not in self.board or self.board[new].state == None:
                                    new_board[new] = CellState("LilyPad")
                            except:
                                continue

                self.board = new_board



def generate_moves(board: dict[Coord, CellState], turn: PlayerColor):
    directions = [Direction.Left, Direction.DownLeft, Direction.Down, Direction.DownRight, Direction.Right]
    if turn == PlayerColor.BLUE:
        for i in range(5):
            directions[i] = -directions[i]
    
    moves = []
    for coord in board:
        if board[coord].state == turn:
            generate_frog_moves(board, coord, directions, moves)
    
    moves.append(GrowAction())
    return moves
    

def generate_frog_moves(board, coord, directions, moves):
    # generate all moves for a frog

    for direc in directions:
        try:
            next_square = coord + direc
            if board[next_square].state == "LilyPad":
                moves.append((MoveAction(coord=coord, _directions=[direc]), next_square))
        except:
            continue
    
    generate_jump_moves(board, coord, directions, set(), list(), coord, moves)

def generate_jump_moves(board: dict[Coord, CellState], coord: Coord, directions: list[Direction],
        visited: set[Coord], path: list[Direction], start_coord: Coord, 
        moves: list[(MoveAction, Coord)]):

    # recursive depth first search for generating jump moves for a given frog

    if coord in visited:
        return 
    
    visited.add(coord)

    for direc in directions:
        try:
            next_square = coord + direc
            next_next_square = next_square + direc
        except:
            continue

        if next_square not in board or next_next_square not in board:
            continue

        if isinstance(board[next_square].state, PlayerColor) and board[next_next_square].state == "LilyPad" \
            and next_next_square not in visited:
            
            new_path = path + [direc]
            moves.append((MoveAction(coord=start_coord, _directions=new_path), next_next_square))
            generate_jump_moves(board, next_next_square, directions, visited, new_path, start_coord, moves)


def simple_eval(board: dict[Coord, CellState]) -> float:
    # function that quickly estimates the evaluation score of a board
    # positive indicates red is winning, negative indicates blue is winning
    red_distance = 0
    blue_distance = 0
    for coord in board:
        match board[coord].state:
            case PlayerColor.RED:
                red_distance += coord.r
            case PlayerColor.BLUE:
                blue_distance += BOARD_N - 1 - coord.r
    
    return red_distance - blue_distance

def minimax(board: dict[Coord, CellState], turn: PlayerColor, depth):
    # red is maximising player, blue is minimising

    if depth == 0:
        return simple_eval(board)