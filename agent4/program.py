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

BOARD_SIZE = BOARD_N * BOARD_N
U64 = np.uint64

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
        
        self.bb = {celltype:np.zeros(1, U64) for celltype in Cell if celltype != Cell.NONE}

        for r in [0,BOARD_N-1]:
            for c in [0,BOARD_N-1]:
                add_cell(self.bb, Cell.LILY,r,c)
        
        for c in range(1,BOARD_N-1):
            add_cell(self.bb, Cell.RED, 0, c)
            add_cell(self.bb, Cell.BLUE, BOARD_N-1, c)
            for r in [1,BOARD_N-2]:
                add_cell(self.bb, Cell.LILY, r, c)
        
        print_board(self.bb)


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        cellcolor = None
        match self._color:
            case PlayerColor.BLUE:
                cellcolor = Cell.BLUE
            case PlayerColor.RED:
                cellcolor = Cell.RED

        eval, move = minimax(self.bb, cellcolor, 5, float('-inf'), float('inf'))
        return move_to_action(self.bb, move)

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

class FrogMove:
    def __init__(self, color, srcbb, dstbb):
        self.color = color
        self.srcbb = srcbb
        self.dstbb= dstbb
        self.prio = abs(int(srcbb).bit_length() - int(dstbb).bit_length())

class GrowMove:
    def __init__(self, color, lilybb):
        self.color = color
        self.lilybb = lilybb
        self.prio = 0.5

def is_inbound(r,c):
    return r < BOARD_N and r >= 0 and c < BOARD_N and c >= 0

def cell_mask(r, c):
    return U64(1) << U64(BOARD_N * r + c)

def row_mask(r):
    return U64(0xFF) << U64(BOARD_N * r)

def col_mask(c):
    mask = U64(0)
    for r in range(BOARD_N):
        mask |= cell_mask(r,c)
    return mask

def direc_mask(bbint, dr, dc):
    shift = BOARD_N * dr + dc
    mask = bbint
    if shift < 0:
        mask >>= U64(-shift)
    else:
        mask <<= U64(shift)
    
    if dr > 0:
        for ddr in range(dr):
            mask &= ~row_mask(ddr)
    if dr < 0:
        for ddr in range(BOARD_N+dr, BOARD_N):
            mask &= ~row_mask(ddr)
    if dc > 0:
        for ddc in range(dc):
            mask &= ~col_mask(ddc)
    
    if dc < 0:
        for ddc in range(BOARD_N+dc, BOARD_N):
            mask &= ~col_mask(ddc)
    
    return mask

def add_cell(bb: dict[Cell, np.ndarray], cell_type: Cell, r, c):
    bb[cell_type][0] |= cell_mask(r,c)

def rm_cell(bb, cell_type, r, c):
    bb[cell_type][0] &= ~cell_mask(r,c)

def is_type(bb, cell_type, r ,c):
    return bb[cell_type][0] & cell_mask(r,c)

def get_type(bb, r, c):
    for celltype in Cell:
        if celltype == Cell.NONE:
            continue 
        if is_type(bb,celltype,r,c):
            return celltype
    return Cell.NONE

def is_occupied(bb, r, c):
    return get_type(bb,r,c) & 2

def get_ones(bbint)->list[U64]:
    ones = []
    while bbint:
        ones.append(bbint & (~bbint+U64(1)))
        bbint &= (bbint-U64(1))
    return ones

def num_ones(bbint):
    tot = 0
    while bbint:
        bbint &= (bbint-U64(1))
        tot += 1
    return tot

def make_move(bb, color, action: Action):
    if isinstance(action, GrowAction):
        bb[Cell.LILY][0] |= grow_mask(bb, color)
        return
    
    src = action.coord
    dst = src

    if isinstance(action._directions, Direction):
        dst = dst + action._directions
    
    else:
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

def print_bb(bbint):
    wtf = {Cell.RED: np.zeros(1,U64), Cell.BLUE: np.zeros(1,U64), Cell.LILY: np.full(1,bbint)}
    print_board(wtf)

def change_color(color):
    return Cell(color ^ 1)

def grow_mask(bb, color):
    lilypad_mask = U64(0)
    for dr in range(-1,2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            lilypad_mask |= direc_mask(bb[color][0], dr, dc)
    
    return lilypad_mask & (~bb[Cell.RED][0]) & (~bb[Cell.BLUE][0] & ~bb[Cell.LILY][0])
    
def print_move(move):
    if isinstance(move, GrowMove):
        print_bb(move.lilybb)
    elif isinstance(move, FrogMove):
        print_bb(move.srcbb | move.dstbb)

def directs(color):
    direcs = [(dr,dc) for dr in range(0,2) for dc in range(-1,2) if (not (dr == 0 and dc == 0))]
    if color == Cell.BLUE:
        for i in range(5):
            dr,dc = direcs[i]
            direcs[i] = (-dr,-dc)
    return direcs

def generate_moves(bb, color):
    moves = []
    directions = directs(color)

    occupiedbb = bb[Cell.RED][0] | bb[Cell.BLUE][0]
    for frogbb in get_ones(bb[color][0]):
        generate_frog_moves(frogbb, color, occupiedbb, 
                            bb[Cell.LILY][0], directions, moves)
    
    moves.append(GrowMove(color, grow_mask(bb, color)))
    return moves
        
def generate_frog_moves(frogbb, color, occupiedbb, lilybb, directions, moves):
    repeat = True
    destbb = frogbb
    while repeat:
        repeat = False
        for dr, dc in directions:
            next = direc_mask(destbb, 2*dr, 2*dc) & lilybb & \
                direc_mask(occupiedbb, dr, dc) & ~destbb
            if next:
                repeat = True
            destbb |= next

    for dr, dc in directions:
        destbb |= (direc_mask(frogbb,dr,dc) & lilybb)
    
    destbb &= ~frogbb
    
    for square in get_ones(destbb):
        moves.append(FrogMove(color, frogbb, square))
    
def move_to_action(bb, move: FrogMove | GrowMove):
    if isinstance(move, GrowMove):
        return GrowAction()

    color = move.color
    directions = directs(color)

    index = int(move.srcbb).bit_length()-1
    src = Coord(index//BOARD_N, index%BOARD_N)
    index = int(move.dstbb).bit_length()-1
    dst = Coord(index//BOARD_N, index%BOARD_N)
    
    for dr,dc in directions:
        dest = direc_mask(move.srcbb, dr, dc)
        if dest & move.dstbb:
            return MoveAction(Coord(*src), Direction(Vector2(dr,dc)))
    
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
    
    def search(board_dict, curr, dest, path, visited):
        if curr == dest:
            return path
        if curr in visited:
            return False
        visited.add(curr)

        for di in directions:
            di = Direction(Vector2(*di))
            try:
                nex = curr + di
                nexnex = nex + di
                if board_dict[nex].state in [PlayerColor.RED, PlayerColor.BLUE] and \
                    board_dict[nexnex].state == 'LilyPad':
                    thepath = search(board_dict, nexnex, dest, path + [di], visited)
                    if thepath != False:
                        return thepath
            except:
                continue
                
        return False

    path = search(board_dict, src, dst, list(), set())
    return MoveAction(src, path)

def apply_move(bb, move):
    if isinstance(move, GrowMove):
        bb[Cell.LILY][0] |= move.lilybb
    
    elif isinstance(move, FrogMove):
        index = int(move.srcbb).bit_length()-1
        src = (index//BOARD_N, index%BOARD_N)
        index = int(move.dstbb).bit_length()-1
        dst = (index//BOARD_N, index%BOARD_N)
        rm_cell(bb, move.color, *src)
        add_cell(bb, move.color, *dst)
        rm_cell(bb, Cell.LILY, *dst)

def undo_move(bb, move):
    if isinstance(move, GrowMove):
        bb[Cell.LILY][0] &= ~move.lilybb
    
    elif isinstance(move, FrogMove):
        index = int(move.srcbb).bit_length()-1
        src = (index//BOARD_N, index%BOARD_N)
        index = int(move.dstbb).bit_length()-1
        dst = (index//BOARD_N, index%BOARD_N)
        rm_cell(bb, move.color, *dst)
        add_cell(bb, move.color, *src)
        add_cell(bb, Cell.LILY, *dst)

def game_over(bb, color):
    match color:
        case Cell.RED:
            return bb[color] == row_mask(7)
        case Cell.BLUE:
            return bb[color] == row_mask(0)
        
def simple_eval(bb) -> float:
    # function that quickly estimates the evaluation score of a board
    # positive indicates red is winning, negative indicates blue is winning
    red_distance = 0
    blue_distance = 0
    
    bbred = bb[Cell.RED][0]
    bbblue = bb[Cell.BLUE][0]
    for square in get_ones(bbred):
        index = int(square).bit_length()-1
        r = index//BOARD_N
        red_distance += r
    
    for square in get_ones(bbblue):
        index = int(square).bit_length()-1
        r = index//BOARD_N
        blue_distance += BOARD_N-1-r
    
    return red_distance - blue_distance

def minimax(bb, color, depth, alpha, beta):
    if depth == 0:
        return simple_eval(bb), None
    
    best_move = None
    best_eval = 0
    moves = generate_moves(bb,color)
    moves.sort(key=lambda x: x.prio)
    if color == Cell.RED: # is maximising player
        best_eval = float('-inf')
        for move in moves:
            apply_move(bb, move)
            nexteval, nextmove = minimax(bb, change_color(color), depth-1, alpha, beta)
            if nexteval > best_eval:
                best_eval = nexteval
                best_move = move
            undo_move(bb, move)
        
            alpha = max(alpha, nexteval)
            if beta <= alpha:
                break

    else:
        best_eval = float('inf')
        for move in moves:
            apply_move(bb, move)
            nexteval, nextmove = minimax(bb, change_color(color), depth-1, alpha, beta)
            if nexteval < best_eval:
                best_eval = nexteval
                best_move = move
            undo_move(bb, move)
        
            beta = min(beta, nexteval)
            if beta <= alpha:
                break
    
    return best_eval, best_move

def iterative_deepening(bb, color,depth):
    