# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from dataclasses import dataclass
from enum import IntEnum
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from referee.game.board import Board, CellState
from referee.game.constants import BOARD_N
HALF_BOARD = (BOARD_N-1)/2
TT_SIZE = 1000003
BIG_NUM = 1000000
NUM_FROGS = 6

import random
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
        
        if color == PlayerColor.RED:
            self._color = Cell.RED 
        else:
            self._color = Cell.BLUE
        
        self.current_color = Cell.RED
        
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
        
        self.time = 0
        self.print_board()
        self.tt = [None] * TT_SIZE
        self.hash = self.get_hash()
        self.last3rows = {Cell.RED: 0, Cell.BLUE: 0}
        self.endrow = {Cell.RED: 0, Cell.BLUE: 0}

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        start_time = time.time()
        best_move = self.iterative_deepening(10, referee['time_remaining'])
        self.time += time.time() - start_time

        print("AGENT bro: ", self.time)
        return move_to_action(best_move)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
    
        self.make_move(action)
        # print_board(self.board)
        # print("ENDROWS: ", self.endrows[Cell.RED], self.endrows[Cell.BLUE])

    def print_board(self):
        board_dict = dict()
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                match self.board[r][c]:
                    case Cell.LILY:
                        board_dict[Coord(r,c)] = CellState('LilyPad')
                    case Cell.BLUE:
                        board_dict[Coord(r,c)] = CellState(PlayerColor.BLUE)
                    case Cell.RED:
                        board_dict[Coord(r,c)] = CellState(PlayerColor.RED)
        
        theboard = Board(board_dict)
        print(theboard.render(use_color=True))

    def grow_set(self):
        surround = [(i,j) for i in range(-1,2) for j in range(-1,2) if not (i == 0 and j==0)]
        lilyset = set()
        for r,c in self.frogs[self.current_color]:
            for dr,dc in surround:
                nr,nc = r+dr, c+dc
                if is_inbound(nr,nc) and self.board[nr][nc] == Cell.NONE:
                    lilyset.add((nr,nc))
        
        return lilyset
    
    def is_occupied(self, r, c):
        return self.board[r][c] & 2
    
    def generate_moves(self):
        moves = []
        directions = get_dirs(self.current_color)

        for r,c in self.frogs[self.current_color]:
            for dr, dc in directions:
                nr,nc = r+dr, c+dc
                if is_inbound(nr,nc) and self.board[nr][nc] == Cell.LILY:
                    moves.append(FrogMove((r,c),(nr,nc),[(dr,dc)]))
            
            self.generate_jump_moves((r,c),directions,moves, set(), list(), (r,c))
        
        moves.append(GrowMove(self.grow_set()))
        return moves
    
    def generate_jump_moves(self, currcoord, directions, moves, visited, path, startcoord):
        if currcoord in visited:
            return
        
        visited.add(currcoord)
        r,c = currcoord
        
        for dr,dc in directions:
            nr,nc = r+dr, c+dc
            nnr,nnc = nr+dr,nc+dc
            if is_inbound(nnr,nnc) and self.is_occupied(nr,nc) and self.board[nnr][nnc] == Cell.LILY:
                newpath = path + [(dr,dc)]
                moves.append(FrogMove(startcoord, (nnr,nnc), newpath))
                self.generate_jump_moves((nnr, nnc), directions, moves, visited, newpath, startcoord)

    def make_move(self, action):
        move = None
        if isinstance(action, GrowAction):
            lilypads = self.grow_set()
            for r,c in lilypads:
                self.board[r][c] = Cell.LILY
            move = GrowMove(lilypads)
        else:
            srccoord = action.coord
            dstcoord = srccoord
            if isinstance(action._directions, Direction):
                dstcoord += action._directions
            else:
                for direct in action._directions:
                    ncoord = dstcoord + direct
                    if self.board[ncoord.r][ncoord.c] == Cell.LILY:
                        dstcoord += direct
                        break
                    else:
                        dstcoord += direct*2
            
            sr,sc = srccoord.r, srccoord.c
            dr,dc = dstcoord.r, dstcoord.c
            self.board[sr][sc] = Cell.NONE
            self.board[dr][dc] = self.current_color
            self.frogs[self.current_color].remove((sr,sc))
            self.frogs[self.current_color].add((dr,dc))
            move = FrogMove((sr, sc), (dr, dc), None)
        
            if (self.current_color == Cell.RED and dr > BOARD_N-3 and sr <= BOARD_N-3) or \
                    (self.current_color == Cell.BLUE and dr < 3 and sr >= 3):
                    self.last3rows[self.current_color] += 1
            
            if (self.current_color == Cell.RED and dr == BOARD_N-1 and sr < BOARD_N-1) or \
                    (self.current_color == Cell.BLUE and dr == 0 and sr > 0):
                    self.endrow[self.current_color] += 1
    
        self.edit_hash(move)
        self.switch_color()

    def simple_eval_side(self, color):
        score = 0
        for r,c in self.frogs[color]:
            score += square_eval(r, c, color)
        return score

    def simple_eval(self):
        return self.simple_eval_side(Cell.RED) - self.simple_eval_side(Cell.BLUE)

    def apply_move(self, move):
        if isinstance(move, GrowMove):
            for r,c in move.lilyset:
                self.board[r][c] = Cell.LILY
        else:
            sr,sc = move.src
            dr,dc = move.dst
            self.board[sr][sc] = Cell.NONE
            self.board[dr][dc] = self.current_color
            self.frogs[self.current_color].remove((sr,sc))
            self.frogs[self.current_color].add((dr,dc))
            
            if (self.current_color == Cell.RED and dr > BOARD_N-3 and sr <= BOARD_N-3) or \
                    (self.current_color == Cell.BLUE and dr < 3 and sr >= 3):
                    self.last3rows[self.current_color] += 1

            if (self.current_color == Cell.RED and dr == BOARD_N-1 and sr < BOARD_N-1) or \
                    (self.current_color == Cell.BLUE and dr == 0 and sr > 0):
                    self.endrow[self.current_color] += 1
        
        self.edit_hash(move)
        self.switch_color()
    
    def undo_move(self, move):
        self.switch_color()
        self.edit_hash(move)

        if isinstance(move, GrowMove):
            for r,c in move.lilyset:
                self.board[r][c] = Cell.NONE
        else:
            sr,sc = move.src
            dr,dc = move.dst
            self.board[dr][dc] = Cell.LILY
            self.board[sr][sc] = self.current_color
            self.frogs[self.current_color].remove((dr,dc))
            self.frogs[self.current_color].add((sr,sc))  

            if (self.current_color == Cell.RED and dr > BOARD_N-3 and sr <= BOARD_N-3) or \
                    (self.current_color == Cell.BLUE and dr < 3 and sr >= 3):
                self.last3rows[self.current_color] -= 1

            if (self.current_color == Cell.RED and dr == BOARD_N-1 and sr < BOARD_N-1) or \
                    (self.current_color == Cell.BLUE and dr == 0 and sr > 0):
                    self.endrow[self.current_color] -= 1

    def null_move(self):
        self.hash ^= (1 << 128)
        self.switch_color()
    
    def switch_color(self):
        if self.current_color == Cell.RED:
            self.current_color = Cell.BLUE
        else:
            self.current_color = Cell.RED

    def minimax(self, depth, alpha, beta, storebest=None) -> float:
        if self.endrow[Cell.RED] == NUM_FROGS:
            return BIG_NUM
        if self.endrow[Cell.BLUE] == NUM_FROGS:
            return -BIG_NUM
        if self.last3rows[Cell.RED] == NUM_FROGS and self.last3rows[Cell.BLUE] == NUM_FROGS:
            reddepth, bluedepth, redstore, bluestore = depth//2, depth//2, None, None
            if self.current_color == Cell.RED:
                reddepth = depth-reddepth
                redstore = storebest
            else:
                bluedepth = depth-bluedepth
                bluestore = storebest
            redeval = self.dfs_eval(Cell.RED, reddepth, 0, redstore)
            blueeval = self.dfs_eval(Cell.BLUE, bluedepth, 0, bluestore)
            return get_dfs_eval(redeval, blueeval, self.current_color)

        index = self.hash % TT_SIZE
        if self.tt[index] != None and self.tt[index].key == self.hash and \
                self.tt[index].depth >= depth:
            if self.tt[index].flag == Flag.EXACT:
                return self.tt[index].val
            if self.tt[index].flag == Flag.LESS and self.tt[index].val <= alpha:
                return alpha
            if self.tt[index].flag == Flag.MORE and self.tt[index].val >= beta:
                return beta
        
        if depth == 0:
            val = self.simple_eval()
            self.tt[index] = TTEntry(self.hash, depth, val, Flag.EXACT, None)
            return val
        
        best_move = None
        best_eval = 0

        moves = self.generate_moves()
        if self.tt[index] != None and self.tt[index].key == self.hash:
            moves.sort(key = lambda x: -prio(x, self.tt[index].move))
        else:
            moves.sort(key= lambda x: -x.prio)

        if self.current_color == Cell.RED:
            best_eval = alpha
            flag = Flag.LESS
            for m in moves:
                self.apply_move(m)
                new_eval = self.minimax(depth-1, alpha, beta)
                self.undo_move(m)
                if new_eval > alpha:
                    alpha = new_eval
                    best_eval = new_eval
                    best_move = m
                    flag = Flag.EXACT

                if beta <= alpha:
                    flag = Flag.MORE
                    break
        else:
            best_eval = beta
            flag = Flag.MORE
            for m in moves:
                self.apply_move(m)
                new_eval = self.minimax(depth-1, alpha, beta)            
                self.undo_move(m)
                if new_eval < beta:
                    beta = new_eval
                    best_eval = new_eval
                    best_move = m
                    flag = Flag.EXACT
                
                if new_eval < beta:
                    beta = new_eval
                if beta <= alpha:
                    flag = Flag.LESS
                    break
        
        if storebest != None:
            storebest.append(best_move)
        
        self.tt[index] = TTEntry(self.hash, depth, best_eval, flag, best_move)
        return best_eval

    def get_hash(self):
        hashed = int(self.current_color)
        for r in range(BOARD_N-1,-1,-1):
            for c in range(BOARD_N-1,-1,-1):
                hashed = (hashed << 2) | self.board[r][c]
        return hashed
        
    def edit_hash(self, move):
        if isinstance(move, GrowMove):
            for r,c in move.lilyset:
                self.hash ^= (Cell.LILY << (2*(BOARD_N * r + c)))
        else:
            sr,sc = move.src
            dr,dc = move.dst
            self.hash ^= (self.current_color << (2*(BOARD_N * sr + sc)))
            self.hash ^= ((self.current_color^Cell.LILY) << (2*(BOARD_N * dr + dc)))
        
        self.hash ^= (1 << 128)

    def iterative_deepening(self, max_depth, time_remaining):
        storedmove = []
        
        start_time = time.time()
        for depth in range(1, max_depth+1):
            storedmove = []
            val = self.minimax(depth, -float("inf"), float("inf"), storebest=storedmove)
            
            if (time.time() - start_time) > time_remaining/30:
                break
            
        
        print("CURR EVAL: ", val)
        print("DEPTH: ", depth)
        return storedmove[0]
    
    def dfs_eval(self, color, depth, ply, storemove=None) -> int:
        if self.endrow[color] == NUM_FROGS:
            return DFS_EVAL(0, True, ply)
        if depth == 0:
            return DFS_EVAL(self.simple_eval_side(color), False)
        
        moves = self.generate_moves()
        moves.sort(key=lambda x: -x.prio)
        best_eval = DFS_EVAL(-float('inf'), False)
        best_move = None
        for m in moves:
            self.apply_move(m)
            self.null_move()
            new_eval = self.dfs_eval(color, depth-1, ply+1)
            self.null_move()
            self.undo_move(m)
            if new_eval > best_eval:
                best_eval = new_eval
                best_move = m
        
        if storemove != None:
            storemove.append(best_move)
        
        return best_eval

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
        sr,sc = src
        dr,dc = dst
        center = (BOARD_N-1)/2
        self.prio = abs(dr-sr) + 0.1 * (max(sr-center,sc-center) - max(dr-center, dc-center))

class GrowMove:
    def __init__(self, lilyset):
        self.lilyset = lilyset
        self.prio = 0.3 * len(lilyset) 

class Flag(IntEnum):
    EXACT = 0
    LESS = 1
    MORE = 2

class TTEntry:
    def __init__(self, key: int, depth, val, flag: Flag, move: FrogMove | GrowMove):
        self.key = key
        self.depth = depth
        self.val = val
        self.flag = flag
        self.move = move

class DFS_EVAL:
    def __init__(self, eval, win, ply=float('inf')):
        self.eval = eval
        self.win = win
        self.ply = ply
    
    def __gt__(self, other):
        if self.win and not other.win:
            return True 
        if not self.win and not other.win and self.eval > other.eval:
            return True
        if self.win and other.win and self.ply < other.ply:
            return True 

        return False

def is_inbound(r,c):
    return r < BOARD_N and r >= 0 and c < BOARD_N and c >= 0

def print_move(move):
    if isinstance(move, GrowMove):
        print(move.lilyset)
    else:
        print(move.src, move.dst)

def get_dirs(color):
    if color == Cell.RED:
        return [(dr,dc) for dr in range(0,2) for dc in range(-1,2) if not (dr == 0 and dc == 0)]
    else:
        return [(dr,dc) for dr in range(-1,1) for dc in range(-1,2) if not (dr == 0 and dc == 0)]

def get_front_dir(color):
    if color == Cell.RED:
        return [(1,dc) for dc in range(-1,2)]
    else:
        return [(-1,dc) for dc in range(-1,2)]

def move_to_action(move: GrowMove | FrogMove):
    if isinstance(move, GrowMove):
        return GrowAction()
    
    return MoveAction(Coord(*move.src), 
        list(map(lambda x: Direction(Vector2(*x)), move.path)))

def switch_color(color):
    if color == Cell.RED:
        return Cell.BLUE
    return Cell.RED

def prio(move, bestmove):
    if bestmove == None:
        return move.prio
    if isinstance(move, GrowMove) and isinstance(bestmove, GrowMove):
        return float("inf")
    if isinstance(move, FrogMove) and isinstance(bestmove, FrogMove) and \
            move.src == bestmove.src and move.dst == bestmove.dst:
        return float("inf")
    return move.prio

def square_eval(r, c, color):
    row_progress = 0
    if color == Cell.RED:
        row_progress = r
    else:
        row_progress = BOARD_N - 1 - r
    
    positional_score = 0.3 * (HALF_BOARD - max(abs(HALF_BOARD - r), abs(HALF_BOARD - c)))
    return row_progress + positional_score

def get_dfs_eval(redeval, blueeval, firstturn):
    if not redeval.win and not blueeval.win:
        return redeval.eval - blueeval.eval 

    if redeval.win and redeval.ply < blueeval.ply:
        return BIG_NUM
    
    if blueeval.win and blueeval.ply < redeval.ply:
        return -BIG_NUM

    if firstturn == Cell.RED:
        return BIG_NUM
    
    return -BIG_NUM
