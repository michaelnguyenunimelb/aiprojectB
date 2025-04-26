# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from enum import IntEnum
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from referee.game.board import Board, CellState
from referee.game.constants import BOARD_N

import random
import time
import numpy as np

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
        
        self.board = np.full((BOARD_N,BOARD_N), Cell.NONE, dtype=np.uint8)

        for r in [0,BOARD_N-1]:
            for c in [0,BOARD_N-1]:
                self.board[r][c] = Cell.LILY
        
        for c in range(1,BOARD_N-1):
            self.board[0][c] = Cell.RED
            self.board[BOARD_N-1][c] = Cell.BLUE
            for r in [1,BOARD_N-2]:
                self.board[r][c] = Cell.LILY


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
        
        moves = generate_moves(self.board,self._color)[0]

        bro = random.choice(moves)
        if isinstance(bro, GrowAction):
            return bro
        return bro[0]

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.

        # start_time = time.time()

        cellcolor = 0
        match color:
            case PlayerColor.BLUE:
                cellcolor = Cell.BLUE
            case PlayerColor.RED:
                cellcolor = Cell.RED

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

                if isinstance(dirs, Direction):
                    curr_coord += dirs
                else: 
                    for di in dirs:
                        point = curr_coord+di
                        if self.board[point.r][point.c] in [Cell.RED, Cell.BLUE]:
                            curr_coord += di * 2
                        elif self.board[point.r][point.c] == Cell.LILY:
                            curr_coord += di
                        else:
                            raise Exception("dawg")
                    
                
                self.board[coord.r][coord.c] = Cell.NONE
                self.board[curr_coord.r][curr_coord.c] = cellcolor
                    
            case GrowAction():
                for r in range(BOARD_N):
                    for c in range(BOARD_N):
                        if self.board[r][c] == cellcolor:
                            for dr in range(-1,2):
                                for dc in range(-1,2):
                                    nr,nc = r+dr, c+dc
                                    if not is_inbound(nr,nc):
                                        continue

                                    if self.board[nr][nc] == Cell.NONE:
                                        self.board[nr][nc] = Cell.LILY


class Cell(IntEnum):
    NONE = 0
    LILY = 1
    RED = 2
    BLUE = 3

def generate_moves(board: np.ndarray, turn: PlayerColor):
    directions = [Direction.Left, Direction.DownLeft, Direction.Down, Direction.DownRight, Direction.Right]
    if turn == PlayerColor.BLUE:
        for i in range(5):
            directions[i] = -directions[i]
    
    cellcolor = 0
    match turn:
        case PlayerColor.BLUE:
            cellcolor = Cell.BLUE
        case PlayerColor.RED:
            cellcolor = Cell.RED
    tim = 0
    moves = []
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board[r][c] == cellcolor:
                start = time.time()
                generate_frog_moves(board, Coord(r,c), directions, moves)
                tim += time.time()-start
    
    moves.append(GrowAction())
    # for m in moves:
    #     board.copy()

    return moves, tim
    

def generate_frog_moves(board, coord, directions, moves):
    # generate all moves for a frog

    for direc in directions:
        try:
            next_square = coord + direc
            if board[next_square.r][next_square.c] == Cell.LILY:
                moves.append((MoveAction(coord=coord, _directions=[direc]), next_square))
        except:
            continue
    
    generate_jump_moves(board, coord, directions, set(), list(), coord, moves)

def generate_jump_moves(board: np.ndarray, coord: Coord, directions: list[Direction],
        visited: set[Coord], path: list[Direction], start_coord: Coord, 
        moves):

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

        if board[next_square.r][next_square.c] & 2 \
            and board[next_next_square.r][next_next_square.c] == Cell.LILY \
            and next_next_square not in visited:
            
            new_path = path + [direc]
            moves.append((MoveAction(coord=start_coord, _directions=new_path), next_next_square))
            generate_jump_moves(board, next_next_square, directions, visited, new_path, start_coord, moves)

def print_board(board):
    board_dict = dict()
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board[r][c] == Cell.LILY:
                board_dict[Coord(r,c)] = CellState('LilyPad')
            elif board[r][c] == Cell.BLUE:
                board_dict[Coord(r,c)] = CellState(PlayerColor.BLUE)
            elif board[r][c] == Cell.RED:
                board_dict[Coord(r,c)] = CellState(PlayerColor.RED)
    
    wtf = Board(board_dict)
    print(wtf.render(use_color=True))
            

def simple_eval(board) -> float:
    # function that quickly estimates the evaluation score of a board
    # positive indicates red is winning, negative indicates blue is winning
    red_distance = 0
    blue_distance = 0
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            match board[r][c]:
                case Cell.RED:
                    red_distance += r
                case Cell.BLUE:
                    blue_distance += BOARD_N-1-r
    
    return red_distance - blue_distance

def minimax(board: dict[Coord, CellState], turn: PlayerColor, depth):
    # red is maximising player, blue is minimising

    if depth == 0:
        return simple_eval(board)

def is_inbound(r,c):
    return r < BOARD_N and r >= 0 and c < BOARD_N and c >= 0
