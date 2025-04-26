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

from referee.game.coord import Vector2

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
        
        self.bb = {celltype: np.uint64 for celltype in Cell if celltype != Cell.NONE}


        for r in [0,BOARD_N-1]:
            for c in [0,BOARD_N-1]:
                add_cell(self.bb, Cell.LILY,r,c)
        
        for c in range(1,BOARD_N-1):
            add_cell(self.bb, Cell.RED, 0, c)
            add_cell(self.bb, Cell.BLUE, BOARD_N-1, c)
            for r in [1,BOARD_N-2]:
                add_cell(self.bb, Cell.LILY, r, c)
        
        print_board(self.bb)
        self._time = 0

        self.gay = 0
        st = time.time()
        for i in range(10000):
            self.bb.copy()
        self.gay += time.time()-st

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")

            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a GROW action")
        
        start = time.time()
        moves = generate_moves(self.bb,self._color)
        self._time += time.time()-start

        print("AGENT 3: ", self._time, self.gay)

        bro = random.choice(moves)
        return bro

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        cellcolor = None
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
        
        make_move(self.bb, cellcolor, action)
        print_board(self.bb)

class Cell(IntEnum):
    NONE = 0
    LILY = 1
    RED = 2
    BLUE = 3

class Move:
    def __init__(self, src, dst, path):
        self.src = src
        self.dst = dst
        self.path = path

def add_cell(bb: dict[Cell, np.ndarray], cell_type: Cell, r, c):
    bb[cell_type][r] |= (1 << c)

def rm_cell(bb, cell_type, r, c):
    bb[cell_type][r] &= ~(1<<c)

def is_type(bb, cell_type, r ,c):
    return (bb[cell_type][r] >> c) & 1

def get_type(bb, r, c):
    for celltype in Cell:
        if celltype == Cell.NONE:
            continue 

        if is_type(bb,celltype,r,c):
            return celltype
    return Cell.NONE

def is_occupied(bb, r, c):
    return get_type(bb,r,c) & 2

def make_move(bb, color, action: Action):

    if isinstance(action, GrowAction):
        for r in range(BOARD_N):
            if not bb[color][r]:
                continue
            for c in range(BOARD_N):
                if not is_type(bb,color,r,c):
                    continue
                for dr in range(-1,2):
                    for dc in range(-1, 2):
                        nr, nc = r+dr, c+dc
                        if not is_inbound(nr,nc):
                            continue
                        if get_type(bb,nr,nc) == Cell.NONE:
                            add_cell(bb,Cell.LILY,nr,nc)
        return
    
    src = action.coord
    dst = src

    for di in action._directions:
        nex = dst+di
        if is_occupied(bb,nex.r,nex.c):
            dst += di*2
        else:
            dst = nex
            break
    
    rm_cell(bb,color,src.r,src.c)
    add_cell(bb, color, dst.r,dst.c)
    rm_cell(bb,Cell.LILY, dst.r,dst.c)

def print_board(bb):
    board_dict = dict()
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            match get_type(bb,r,c):
                case Cell.LILY:
                    board_dict[Coord(r,c)] = CellState('LilyPad')
                case Cell.BLUE:
                    board_dict[Coord(r,c)] = CellState(PlayerColor.BLUE)
                case Cell.RED:
                    board_dict[Coord(r,c)] = CellState(PlayerColor.RED)
    
    wtf = Board(board_dict)
    print(wtf.render(use_color=True))

def generate_moves(bb: np.ndarray, turn: PlayerColor):
    directions = [(dr,dc) for dr in range(0,2) for dc in range(-1,2) if not (dr == 0 and dc == 0)]
    if turn == PlayerColor.BLUE:
        for i in range(5):
            dr,dc = directions[i]
            directions[i] = (-dr,-dc)
    
    cellcolor = 0
    match turn:
        case PlayerColor.BLUE:
            cellcolor = Cell.BLUE
        case PlayerColor.RED:
            cellcolor = Cell.RED
    
    moves = []
    for r in range(BOARD_N):
        row = bb[cellcolor][r]
        if not row:
            continue

        for c in range(BOARD_N):
            if is_type(bb,cellcolor,r,c):
                generate_frog_moves(bb, (r, c), directions, moves)
    
    moves.append(GrowAction())

    return moves
    
def generate_frog_moves(bb, src: tuple[int], directions: list[tuple[int]], moves):
    # generate all moves for a frog
    r,c = src
    for direc in directions:
        dr, dc = direc
        nexr, nexc = r + dr, c + dc
        if is_inbound(nexr, nexc) and is_type(bb, Cell.LILY, nexr, nexc):
            moves.append(MoveAction(coord=Coord(r,c), _directions=[Direction(Vector2(dr,dc))]))
    
    generate_jump_moves(bb, (r, c), directions, set(), list(), (r, c), moves)

def generate_jump_moves(bb, coord, directions,
        visited, path, src, 
        moves):

    # recursive depth first search for generating jump moves for a given frog
    if coord in visited:
        return
    
    visited.add(coord)

    r,c = coord
    for direc in directions:
        dr, dc = direc
        nexr, nexc = r+dr, c+dc
        nexnexr,nexnexc = nexr+dr, nexc+dc
        
        if (not is_inbound(nexnexr,nexnexc)) or (nexnexr,nexnexc) in visited:
            continue

        if is_occupied(bb,nexr,nexc) and \
            is_type(bb, Cell.LILY, nexnexr, nexnexc):
            
            new_path = path + [(dr,dc)]
            moves.append(MoveAction(coord=Coord(*src), _directions=new_path))
            generate_jump_moves(bb, (nexnexr,nexnexc), directions, visited, new_path, src, moves)

def simple_eval(bb) -> float:
    # function that quickly estimates the evaluation score of a board
    # positive indicates red is winning, negative indicates blue is winning
    red_distance = 0
    blue_distance = 0
    for r in range(BOARD_N):
        red = bb[Cell.RED][r]
        blue = bb[Cell.BLUE][r]
        while red:
            red &= red-1
            red_distance += r
        while blue:
            blue &= blue-1
            blue_distance += BOARD_N-1-r
    
    return red_distance - blue_distance

def minimax(board: dict[Coord, CellState], turn: PlayerColor, depth):
    # red is maximising player, blue is minimising

    if depth == 0:
        return simple_eval(board)
    
def is_inbound(r,c):
    return r < BOARD_N and r >= 0 and c < BOARD_N and c >= 0
