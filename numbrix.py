# Grupo 74:
# 95536 Sebastião Assunção
# 95638 Martim Santos

import sys
import os
# import psutil
from tokenize import String
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_graph_search, depth_first_tree_search, greedy_search, recursive_best_first_search, uniform_cost_search, InstrumentedProblem
import cProfile
import pstats

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

class Board:
    """ Representação interna de um tabuleiro de Numbrix. """

    def __init__(self, board, n, positions, missing_numbers):
        self.grid = board
        self.n = n
        self.missing_numbers = missing_numbers
        self.positions = positions

    def get_number(self, row: int, col: int):
        """ Devolve o valor na respetiva posição do tabuleiro. """
        if 0 <= row <= self.n-1 and 0 <= col <= self.n-1:
            return self.grid[row][col]
        else:
            return None

    def adjacent_vertical_numbers(self, row: int, col: int):
        """ Devolve os valores imediatamente abaixo e acima,
        respectivamente. """
        return (self.get_number(row+1, col), self.get_number(row-1, col))

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        return (self.get_number(row, col-1), self.get_number(row, col+1))

    def adjacent_positions(self, row: int, col: int):
        """ Devolve as posições adjacentes livres do tabuleiro. """
        positions = []
        down, up = self.adjacent_vertical_numbers(row, col)
        left, right = self.adjacent_horizontal_numbers(row, col)
        if down != None:
            positions += [(row+1, col)]
        if up != None:
            positions += [(row-1, col)]
        if left != None:
            positions += [(row, col-1)]
        if right != None:
            positions += [(row, col+1)]
        return positions

    def free_adjacent_positions(self, row: int, col: int):
        """ Devolve as posições adjacentes livres do tabuleiro. """
        positions = []
        down, up = self.adjacent_vertical_numbers(row, col)
        left, right = self.adjacent_horizontal_numbers(row, col)
        if down == 0:
            positions += [(row+1, col)]
        if up == 0:
            positions += [(row-1, col)]
        if left == 0:
            positions += [(row, col-1)]
        if right == 0:
            positions += [(row, col+1)]
        return positions

    def manhattan_distance(self, pos1, pos2):
        """Retorna a distancia de manhattan entre pos1 e pos2"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def valid_manhattan_distance(self, pos: tuple, num: int) -> bool:
        """ Returns True if for all the placed numbers on the board, the numerical diference between
        the number on that position and each numbers position is greater than the manhattan distance between
        that position and each numbers position. """
        
        next_num, prev_num = None, None

        for number in range(num + 1, self.n**2+1):
            if self.positions[number-1] != None:
                next_num = number
                break

        for number in range(num - 1, 0, -1):
            if self.positions[number-1] != None:
                prev_num = number
                break

        reach_next, reach_prev = False, False

        reach_prev, reach_next = True, True

        if prev_num != None and next_num == None:
            reach_prev = self.manhattan_distance(pos, self.positions[prev_num-1]) <= abs (num - prev_num)

        elif next_num != None and prev_num == None:
            reach_next = self.manhattan_distance(pos, self.positions[next_num-1]) <= abs (num - next_num)

        elif next_num != None and prev_num != None:
            reach_prev = self.manhattan_distance(pos, self.positions[prev_num-1]) <= abs (num - prev_num)
            reach_next = self.manhattan_distance(pos, self.positions[next_num-1]) <= abs (num - next_num)

        return reach_prev and reach_next

    def possible_positions(self, num: int):
        """ Devolve as posições possíveis para colocar o número num. """

        pos_prev = self.positions[num-2] if num > 1 else None
        pos_next = self.positions[num] if num < self.n**2 else None

        if pos_prev is None and pos_next is None:
            return None
        elif pos_prev != None and pos_next is None:
            return self.free_adjacent_positions(pos_prev[0], pos_prev[1])
        elif pos_prev is None and pos_next != None:
            return self.free_adjacent_positions(pos_next[0], pos_next[1])
        else:
            return list(set(self.free_adjacent_positions(pos_prev[0], pos_prev[1]))
                        .intersection(set(self.free_adjacent_positions(pos_next[0], pos_next[1]))))

    def valid_position(self, pos: tuple, num: int) -> bool:

        def isolated_zero(pos: tuple, zero: int, possible_num: int):
            if zero == 0:
                adjacent_numbers = self.adjacent_vertical_numbers(pos[0], pos[1]) \
                            + self.adjacent_horizontal_numbers(pos[0], pos[1]) + (possible_num, )
                
                # Se tiver mais do que 1 zero (o adjacente que será colocado)
                # então não será um zero isolado
                if adjacent_numbers.count(0) > 1:
                    return False

                # Filtrar os None ou os 0's
                placed_adj_numbers = [num for num in adjacent_numbers if (num != None and num != 0)]

                # Verificar se há dois numeros que precisem desse zero (isolado mas será usado)
                for i, num in enumerate(placed_adj_numbers):
                    
                    # Casos onde o zero pode estar isolado e so precisa de um adjacente
                    if (num == self.n**2-1 and self.positions[self.n**2-1] == None):
                        return False
                    elif num == 2 and self.positions[0] == None:
                        return False

                    for num2 in placed_adj_numbers[i+1:]:
                        if abs(num-num2) == 2:
                            if self.positions[max(num, num2)-2] == None:
                                return False
                # Ou não há numeros com 2 de diferença ou os que existem já têm o numero do meio
                return True

            return False

        def valid_adjacent(pos: tuple, num: int):
                
            if num == 0:     
                return True

            adj_zeros = len(self.free_adjacent_positions(pos[0], pos[1]))
            if (num == 1):
                missing_adj = 1 if 2 in self.missing_numbers else 0
            elif (num == self.n**2):
                missing_adj = 1 if self.n**2 - 1 in self.missing_numbers else 0
            else:
                missing_adj = 0
                if self.positions[num-2] is None:
                    missing_adj += 1
                if self.positions[num] is None:
                    missing_adj += 1

            if (missing_adj > adj_zeros):
                return False

            return True

        if valid_adjacent(pos, num):

            adjacent_positions = self.adjacent_positions(pos[0], pos[1])

            for row, col in adjacent_positions:
                if self.get_number(row, col) != None and (not valid_adjacent((row,col), self.get_number(row, col)) \
                                                    or isolated_zero((row,col), self.get_number(row, col), num)):
                    return False
            return True

        else:
            return False

    def to_string(self) -> String:
        return '\n'.join(['\t'.join([str(item) for item in row]) for row in self.grid])

    @ staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """
        with open(filename, "r") as input_file:
            f = input_file.readline()
            n = int(f)

            a = input_file.read()
            input_file.close()
        a = a.split("\n")[:-1]

        grid = [[int(n) for n in row.split("\t")] for row in a]

        placed_numbers = list()
        positions = [None] * n**2

        # Generate placed numbers
        for i, row in enumerate(grid):
            for j, num in enumerate(row):
                if num != 0:
                    placed_numbers.append(num)
                    positions[num-1] = (i, j)
        placed_numbers = sorted(placed_numbers)

        # Complete missing numbers
        missing_numbers = list()
        for i in range(1, n**2+1):
            if i not in placed_numbers:
                missing_numbers.append(i)

        return Board(grid, n, positions, missing_numbers)


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        self.initial = NumbrixState(board)

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """

        # Escolher o melhor numero para colocar no tabuleiro
        best = None
        best_pos = []
        min_best_pos = float('inf')
        for num in state.board.missing_numbers:

            possible_pos = state.board.possible_positions(num)

            if possible_pos is None:
                continue

            # Filter possible_positions by the ones that dont violate future manhattan distance constraints and dont block neighbouring numbers
            if valid_possible_pos := [pos for pos in possible_pos if (state.board.valid_manhattan_distance(pos, num) and state.board.valid_position(pos, num))]:

                # If there is only one valid position, choose it
                if len(valid_possible_pos) == 1:
                    return [valid_possible_pos[0] + (num, )]

                # Choose the position that minimizes the number of possible positions
                if len(valid_possible_pos) < min_best_pos:
                    best = num
                    best_pos = valid_possible_pos
                    min_best_pos = len(valid_possible_pos)

            else:
                return []
        
        return [] if best is None else [pos + (best, ) for pos in best_pos]

    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """

        pos_x, pos_y, num = action

        new_grid = list()
        for row in state.board.grid:
            new_row = list()
            for el in row:
                new_row.append(el)
            new_grid.append(new_row)
        
        new_grid[pos_x][pos_y] = num
        
        new_positions = list()
        for pos in state.board.positions:
            new_positions.append(pos)
        new_positions[num-1] = (pos_x, pos_y)

        new_missing_numbers = list()
        for number in state.board.missing_numbers:
            new_missing_numbers.append(number)
        new_missing_numbers.remove(num)

        board = Board(new_grid, state.board.n, new_positions, new_missing_numbers)

        return NumbrixState(board)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes. """
        
        if state.board.missing_numbers:
            return False

        for i, row in enumerate(state.board.grid):
            for j, element in enumerate(row):
                adjacents = state.board.adjacent_vertical_numbers(i, j) \
                    + state.board.adjacent_horizontal_numbers(i, j)
                if ((
                    element + 1 <= state.board.n ** 2
                    and element + 1 not in adjacents
                )
                    or element > 1
                    and element - 1 not in adjacents
                ):
                    return False

        return True

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura greedy. """

        board = node.state.board
        action = node.action

        zeros = len(board.missing_numbers)

        penalty = 0

        # DFS para encontrar ilhas de 0's
        def DFS(i, j, end, max_isle_size, visited):

            adjacents = board.adjacent_vertical_numbers(i, j) \
                        + board.adjacent_horizontal_numbers(i, j)

            if i < 0 or i >= board.n or j < 0 or j >= board.n or board.grid[i][j] != 0 \
                or adjacents.count(0) > 2 or (i, j) in visited or max_isle_size == 0 or end == True:
                return

            # marcar como 0 visitado
            visited.append((i,j))

            if (adjacents.count(0) == 1):
                end = True

            max_isle_size -= 1

            # Ir aos 4 adjacentes
            DFS(i - 1, j, end, max_isle_size, visited)
            DFS(i, j - 1, end, max_isle_size, visited)
            DFS(i, j + 1, end, max_isle_size, visited)
            DFS(i + 1, j, end, max_isle_size, visited)

        if action:
            pos_x, pos_y, _ = action

            adjacents = board.adjacent_vertical_numbers(pos_x, pos_y) + \
                    board.adjacent_horizontal_numbers(pos_x, pos_y)
            
            adjacent_positions = board.adjacent_positions(pos_x, pos_y)

            penalty = adjacents.count(0)**2

            max_isle_size = 2
            end = False
            visited = []
            
            for pos in adjacent_positions:
                if board.get_number(pos_x, pos_y) == 0:
                    adj_0 = board.adjacent_vertical_numbers(pos[0], pos[1]) + \
                    board.adjacent_horizontal_numbers(pos[0], pos[1])
                    if adj_0.count(0) <= 2:
                        DFS(pos_x, pos_y, end, max_isle_size, visited)
                    if end == True:
                        return float('inf')

            for adj in adjacents:
                if adj != 0 and adj != None:
                    pos = board.positions[adj-1]
                    adj_adj = board.adjacent_vertical_numbers(pos[0], pos[1]) + \
                                board.adjacent_horizontal_numbers(pos[0], pos[1])
                    penalty += adj_adj.count(0)
                else:
                    penalty += 1
    
        return zeros + 1.3*penalty


if __name__ == "__main__":

    # Ler o ficheiro de input de sys.argv[1],
    # process = psutil.Process(os.getpid())
    board = Board.parse_instance(sys.argv[1])
    problem = Numbrix(board)
    # with cProfile.Profile() as profile:
    goal_node = greedy_search(problem)
        #stats = pstats.Stats(profile)
        #stats.sort_stats(pstats.SortKey.TIME)
        #stats.print_stats()
    #goal_node = depth_first_tree_search(problem)
    # memoria = process.memory_info().rss // 1024
    #print("Teste: ", sys.argv[1])
    #print(f'Memoria utilizada: {memoria} KB')
    #print(f'Numero de nos gerados: {problem.states}')
    #print(f'Numero de nos expandidos: {problem.succs}')
    
    print(goal_node.state.board.to_string(),
          sep="") if goal_node else print("Solução não encontrada")
