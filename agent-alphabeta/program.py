# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from dataclasses import dataclass
from enum import IntEnum
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from referee.game.board import Board, CellState
from referee.game.constants import BOARD_N

import time

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
        
        self.board = [[Cell.NONE for _ in range(BOARD_N)] for _ in range(BOARD_N)]
        self.frogs = {Cell.RED: set(), Cell.BLUE: set()}

        for r in [0,BOARD_N-1]:
            for c in [0,BOARD_N-1]:
                self.board[r][c] = Cell.LILY
        
        for c in range(1,BOARD_N-1):
            self.board[0][c]= Cell.RED
            self.board[BOARD_N - 1][c] = Cell.BLUE
            self.frogs[Cell.RED].add((0,c))
            self.frogs[Cell.BLUE].add((BOARD_N-1,c))

            for r in [1,BOARD_N-2]:
                self.board[r][c] = Cell.LILY
        
        print_board(self.board)

        self.time = 0


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

        storedmove = []

        start_time = time.time()
        minimax(self.board, self.frogs, cellcolor, 5, -float("inf"), float("inf"),storedmove)
        self.time += time.time()-start_time

        print("AGENT AB: ", self.time)
        return move_to_action(storedmove[0])

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
    
        make_move(self.board, self.frogs, cellcolor, action)
        print_board(self.board)

class Cell(IntEnum):
    NONE = 0
    LILY = 1
    RED = 2
    BLUE = 3

class FrogMove:
    def __init__(self, src, dst, path):
        self.src = src
        self.dst = dst
        self.path = path

class GrowMove:
    def __init__(self, lilyset):
        self.lilyset = lilyset
        

def is_inbound(r,c):
    return r < BOARD_N and r >= 0 and c < BOARD_N and c >= 0

def print_board(board):
    board_dict = dict()
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            match board[r][c]:
                case Cell.LILY:
                    board_dict[Coord(r,c)] = CellState('LilyPad')
                case Cell.BLUE:
                    board_dict[Coord(r,c)] = CellState(PlayerColor.BLUE)
                case Cell.RED:
                    board_dict[Coord(r,c)] = CellState(PlayerColor.RED)
    
    theboard = Board(board_dict)
    print(theboard.render(use_color=True))

def print_move(move):
    if isinstance(move, GrowMove):
        print(move.lilyset)
    else:
        print(move.src, move.dst)

def grow_set(board, frogs, color):
    surround = [(i,j) for i in range(-1,2) for j in range(-1,2) if not (i == 0 and j==0)]
    lilyset = set()
    for r,c in frogs[color]:
        for dr,dc in surround:
            nr,nc = r+dr, c+dc
            if is_inbound(nr,nc) and board[nr][nc] == Cell.NONE:
                lilyset.add((nr,nc))
    
    return lilyset

def get_dirs(color):
    if color == Cell.RED:
        return [(dr,dc) for dr in range(0,2) for dc in range(-1,2) if not (dr == 0 and dc == 0)]
    else:
        return [(dr,dc) for dr in range(-1,1) for dc in range(-1,2) if not (dr == 0 and dc == 0)]

def is_occupied(board, r, c):
    return board[r][c] & 2

def generate_moves(board, frogs, color):
    moves = []
    directions = get_dirs(color)

    for r,c in frogs[color]:
        for dr, dc in directions:
            nr,nc = r+dr, c+dc
            if is_inbound(nr,nc) and board[nr][nc] == Cell.LILY:
                moves.append(FrogMove((r,c),(nr,nc),[(dr,dc)]))
        
        generate_jump_moves(board,(r,c),directions,moves, set(), list(), (r,c))
    
    moves.append(GrowMove(grow_set(board, frogs, color)))
    return moves

def generate_jump_moves(board, currcoord, directions, moves, visited, path, startcoord):
    if currcoord in visited:
        return
    
    visited.add(currcoord)
    r,c = currcoord
    
    for dr,dc in directions:
        nr,nc = r+dr, c+dc
        nnr,nnc = nr+dr,nc+dc
        if is_inbound(nnr,nnc) and is_occupied(board, nr,nc) and board[nnr][nnc] == Cell.LILY:
            newpath = path + [(dr,dc)]
            moves.append(FrogMove(startcoord, (nnr,nnc), newpath))
            generate_jump_moves(board, (nnr, nnc), directions, moves, visited, newpath, startcoord)

def move_to_action(move: GrowMove | FrogMove):
    if isinstance(move, GrowMove):
        return GrowAction()
    
    return MoveAction(Coord(*move.src), 
        list(map(lambda x: Direction(Vector2(*x)), move.path)))

def make_move(board, frogs, color, action):
    if isinstance(action, GrowAction):
        lilypads = grow_set(board, frogs, color)
        for r,c in lilypads:
            board[r][c] = Cell.LILY
    
    else:
        srccoord = action.coord
        dstcoord = srccoord
        if isinstance(action._directions, Direction):
            dstcoord += action._directions
        else:
            for direct in action._directions:
                ncoord = dstcoord + direct
                if board[ncoord.r][ncoord.c] == Cell.LILY:
                    dstcoord += direct
                    break
                else:
                    dstcoord += direct*2
        
        sr,sc = srccoord.r, srccoord.c
        dr,dc = dstcoord.r, dstcoord.c
        board[sr][sc] = Cell.NONE
        board[dr][dc] = color
        frogs[color].remove((sr,sc))
        frogs[color].add((dr,dc))

def simple_eval_side(board, frogs, color):
    progress_score = 0
    positional_score = 0
    for r,c in frogs[color]:
        if color == Cell.RED:
            progress_score += r
        else:
            progress_score += BOARD_N - 1 - r
        positional_score += -0.2 * abs(c- (BOARD_N-1)/2)   # favours positions in center
    
    return progress_score + positional_score

def simple_eval(board, frogs):
    return simple_eval_side(board, frogs, Cell.RED) - simple_eval_side(board, frogs, Cell.BLUE)

def apply_move(board, frogs, color, move):
    if isinstance(move, GrowMove):
        for r,c in move.lilyset:
            board[r][c] = Cell.LILY
    else:
        sr,sc = move.src
        dr,dc = move.dst
        board[sr][sc] = Cell.NONE
        board[dr][dc] = color
        frogs[color].remove((sr,sc))
        frogs[color].add((dr,dc))

def undo_move(board, frogs, color, move):
    if isinstance(move, GrowMove):
        for r,c in move.lilyset:
            board[r][c] = Cell.NONE
    else:
        sr,sc = move.src
        dr,dc = move.dst
        board[dr][dc] = Cell.LILY
        board[sr][sc] = color
        frogs[color].remove((dr,dc))
        frogs[color].add((sr,sc))
        
def switch_color(color):
    if color == Cell.RED:
        return Cell.BLUE
    return Cell.RED

def minimax(board, frogs, color, depth, alpha, beta, storebest=None):
    if depth == 0:
        return simple_eval(board, frogs)
    
    best_move = None
    best_eval = 0

    moves = generate_moves(board,frogs,color)

    if color == Cell.RED:
        best_eval = -float("inf")
        for m in moves:
            apply_move(board, frogs, color, m)
            new_eval = minimax(board, frogs, switch_color(color), depth-1, alpha, beta)
            undo_move(board, frogs, color, m)
            if new_eval > best_eval:
                best_eval = new_eval
                best_move = m
            
            alpha = max(alpha, new_eval)
            if beta <= alpha:
                break
    else:
        best_eval = float("inf")
        for m in moves:
            apply_move(board, frogs, color, m)
            new_eval = minimax(board, frogs, switch_color(color), depth-1, alpha, beta)
            undo_move(board, frogs, color, m)
            if new_eval < best_eval:
                best_eval = new_eval
                best_move = m
            
            beta = min(beta, new_eval)
            if beta <= alpha:
                break
    
    if storebest != None:
        storebest.append(best_move)
    
    return best_eval