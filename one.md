PRACTICAL NO-1
1A. Write a program to implement depth first search algorithm. 
graph1 = {
 'A': set(['B', 'C']),
 'B': set(['A', 'D', 'E']),
 'C': set(['A', 'F']),
 'D': set(['B']),
 'E': set(['B', 'F']),
 'F': set(['C', 'E'])
 }
def dfs(graph, node, visited):
 if node not in visited:
     visited.append(node)
     for n in graph[node]:
         dfs(graph,n, visited)
 return visited
visited = dfs(graph1,'A', [])
print(visited)


1B. Write a program to implement breadth first search algorithm
graph = {'A': set(['B', 'C']),
 'B': set(['A', 'D', 'E']),
 'C': set(['A', 'F']),
 'D': set(['B']),
 'E': set(['B', 'F']),
 'F': set(['C', 'E'])
 }
#Implement Logic of BFS
def bfs(start):
 queue = [start]
 levels={} #This Dict Keeps track of levels
 levels[start]=0 #Depth of start node is 0
 visited = set(start)
 while queue:
     node = queue.pop(0)
     neighbours=graph[node]
     for neighbor in neighbours:
         if neighbor not in visited:
             queue.append(neighbor)
             visited.add(neighbor)
             levels[neighbor]= levels[node]+1
 print(levels) #print graph level
 return visited
print(str(bfs('A'))) #print graph node

#For Finding Breadth First Search Path
def bfs_paths(graph, start, goal):
 queue = [(start, [start])]
 while queue:
     (vertex, path) = queue.pop(0)
     for next in graph[vertex] - set(path):
         if next == goal:
             yield path + [next]
         else:
             queue.append((next, path + [next]))
result=list(bfs_paths(graph, 'A', 'F'))
print(result)# [['A', 'C', 'F'], ['A', 'B', 'E', 'F']]

#For finding shortest path
def shortest_path(graph, start, goal):
 try:
     return next(bfs_paths(graph, start, goal))
 except StopIteration:
     return None
result1=shortest_path(graph, 'A', 'F')
print(result1)# ['A', 'C', 'F']


2A. Write a program to simulate 4-Queen / N-Queen problem.
class QueenChessBoard:
    def __init__(self, size):
        self.size = size
        self.columns = []
    def place_in_next_row(self, column):
        self.columns.append(column)
    def remove_in_current_row(self):
        return self.columns.pop()
    def is_this_column_safe_in_next_row(self, column):
        # index of next row
        row = len(self.columns)
        # check column
        for queen_column in self.columns:
            if column == queen_column:
                return False
            # check diagonal
        for queen_row, queen_column in enumerate(self.columns):
            if queen_column - queen_row == column - row:
                return False
        # check other diagonal
        for queen_row, queen_column in enumerate(self.columns):
            if ((self.size - queen_column) - queen_row== (self.size - column) - row):
                return False
        return True
    def display(self):
        for row in range(self.size):
            for column in range(self.size):
                if column == self.columns[row]:
                    print('Q', end=' ')
                else:
                    print('.', end=' ')
            print()
def solve_queen(size):
#To display the chess board
    board = QueenChessBoard(size)
    number_of_solutions = 0
    row = 0
    column = 0
    # iterate over rows of board
    while True:
        while column < size:
            if board.is_this_column_safe_in_next_row(column):
                board.place_in_next_row(column)
                row += 1
                column = 0
                break
            else:
                column += 1
        if (column == size or row == size):
            if row == size:
                board.display()
                print()
                number_of_solutions += 1
                board.remove_in_current_row()
                row -= 1
            try:
                prev_column = board.remove_in_current_row()
            except IndexError:
                break
            row -= 1
            column = 1 + prev_column
    print('Number of solutions:', number_of_solutions)
n = int(input('Enter n: '))
solve_queen(n)


2B. Write a program to solve tower of Hanoi problem
def moveTower(height,fromPole, toPole, withPole):
 if height >= 1:
     moveTower(height-1,fromPole,withPole,toPole)
     moveDisk(fromPole,toPole)
     moveTower(height-1,withPole,toPole,fromPole)
def moveDisk(fp,tp):
 print("moving disk from",fp,"to",tp)
moveTower(3,"A","B","C")


3A. Write a program to implement alpha beta search.
tree = [[[5, 1, 2], [8, -8, -9]], [[9, 4, 5], [-3, 4, 3]]]
root = 0
pruned = 0
def children(branch, depth, alpha, beta):
    global tree
    global root
    global pruned
    i = 0
    for child in branch:
        if type(child) is list:
            (nalpha, nbeta) = children(child, depth + 1, alpha, beta)
            if depth % 2 == 1:
                beta = nalpha if nalpha < beta else beta
            else:
                alpha = nbeta if nbeta > alpha else alpha
                branch[i] = alpha if depth % 2 == 0 else beta
                i += 1
        else:
            if depth % 2 == 0 and alpha < child:
                alpha = child
            if depth % 2 == 1 and beta > child:
                beta = child
            if alpha >= beta:
                pruned += 1
                break
    if depth == root:
        tree = alpha if root == 0 else beta
    return (alpha, beta)
def alphabeta(in_tree=tree, start=root, upper=-15, lower=15):
 global tree

 global pruned
 global root
 (alpha, beta) = children(tree, start, upper, lower)

 if __name__ == "__main__":
     print ("(alpha, beta): ", alpha, beta)
     print ("Result: ", tree)
     print ("Times pruned: ", pruned)
 return (alpha, beta, tree, pruned)
if __name__ == "__main__":
 alphabeta(None)


3B. Write a program for hill climbing problem
import math
increment = 0.1
startingPoint = [1, 1]
point1 = [1,5]
point2 = [6,4]
point3 = [5,2]
point4 = [2,1]
def distance(x1, y1, x2, y2):
    dist = math.pow(x2-x1, 2) + math.pow(y2-y1, 2)
    return dist
def sumOfDistances(x1, y1, px1, py1, px2, py2, px3, py3, px4, py4):
 d1 = distance(x1, y1, px1, py1)
 d2 = distance(x1, y1, px2, py2)
 d3 = distance(x1, y1, px3, py3)
 d4 = distance(x1, y1, px4, py4)
 return d1 + d2 + d3 + d4

def newDistance(x1, y1, point1, point2, point3, point4):
 d1 = [x1, y1]
 d1temp = sumOfDistances(x1, y1, point1[0],point1[1], point2[0],point2[1],point3[0],point3[1], point4[0],point4[1] )
 d1.append(d1temp)
 return d1

minDistance = sumOfDistances(startingPoint[0], startingPoint[1],point1[0],point1[1], point2[0],point2[1],
 point3[0],point3[1], point4[0],point4[1] )
flag = True
def newPoints(minimum, d1, d2, d3, d4):
 if d1[2] == minimum:
     return [d1[0], d1[1]]
 elif d2[2] == minimum:
     return [d2[0], d2[1]]
 elif d3[2] == minimum:
     return [d3[0], d3[1]]
 elif d4[2] == minimum:
     return [d4[0], d4[1]]
i = 1
while flag:
 d1 = newDistance(startingPoint[0]+increment, startingPoint[1], point1, point2,point3, point4)
 d2 = newDistance(startingPoint[0]-increment, startingPoint[1], point1, point2,point3, point4)
 d3 = newDistance(startingPoint[0], startingPoint[1]+increment, point1, point2,point3, point4)
 d4 = newDistance(startingPoint[0], startingPoint[1]-increment, point1, point2,point3, point4)
 print (i,' ', round(startingPoint[0], 2), round(startingPoint[1], 2))
 minimum = min(d1[2], d2[2], d3[2], d4[2])
 if minimum < minDistance:
     startingPoint = newPoints(minimum, d1, d2, d3, d4)
     minDistance = minimum
     i+=1
 else:
     flag = False


4A. Implement A* Algorithm
Create a script folder and open it in cmd
pip install simpleai
pip install pydot flask
from simpleai.search import SearchProblem, astar
GOAL = 'HELLO WORLD'
class HelloProblem(SearchProblem):
    def actions(self, state):
        if len(state) < len(GOAL):
            return list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            return []
    def result(self, state, action):
        return state + action

    def is_goal(self, state):
        return state == GOAL
    def heuristic(self, state):
        wrong = sum([1 if state[i] != GOAL[i] else 0
                     for i in range(len(state))])
        missing = len(GOAL) - len(state)
        return wrong + missing
problem = HelloProblem(initial_state='')
result = astar(problem)
print(result.state)
print(result.path())


4B. Write  a program to solve water jug problem
capacity = (12,8,5)
x = capacity[0]
y = capacity[1]
z = capacity[2]
memory = {}
ans = []
def get_all_states(state):
 a = state[0]
 b = state[1]
 c = state[2]
 if(a==6 and b==6):
     ans.append(state)
     return True
 if((a,b,c) in memory):
     return False
 memory[(a,b,c)] = 1
 if(a>0):
     if(a+b<=y):
         if( get_all_states((0,a+b,c)) ):
             ans.append(state)
             return True
     else:
        if( get_all_states((a-(y-b), y, c)) ):
            ans.append(state)
            return True
 #empty a into c
 if(a+c<=z):
        if( get_all_states((0,b,a+c)) ):
            ans.append(state)
            return True
        else:
            if( get_all_states((a-(z-c), b, z)) ):
                ans.append(state)
                return True
 #empty jug b
 if(b>0):
      if(a+b<=x):
          if( get_all_states((a+b, 0, c)) ):
              ans.append(state)
              return True
      else:
          if( get_all_states((x, b-(x-a), c)) ):
              ans.append(state)
              return True
 #empty b into c
      if(b+c<=z):
          if( get_all_states((a, 0, b+c)) ):
              ans.append(state)
              return True
      else:
          if( get_all_states((a, b-(z-c), z)) ):
                  ans.append(state)
                  return True
 #empty jug c
 if(c>0):
      if(a+c<=x):
         if( get_all_states((a+c, b, 0)) ):
             ans.append(state)
             return True
      else:
         if( get_all_states((x, b, c-(x-a))) ):
             ans.append(state)
             return True
 #empty c into b
 if(b+c<=y):
     if( get_all_states((a, b+c, 0)) ):
         ans.append(state)
         return True
     else:
         if( get_all_states((a, y, c-(y-b))) ):
             ans.append(state)
             return True
 return False

initial_state = (12,0,0)
print("Starting work...\n")
get_all_states(initial_state)
ans.reverse()
for i in ans:
 print(i)


5A. Simulate tic-tac-toe game using min-max algorithm
import math
# Board setup
board = [
    [' ', ' ', ' '],
    [' ', ' ', ' '],
    [' ', ' ', ' ']
]

# Check if there are any empty spaces on the board
def is_moves_left(board):
    for row in board:
        if ' ' in row:
            return True
    return False

# Function to evaluate the board to check for a win
def evaluate(board):
    # Check rows for victory
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2]:
            if board[row][0] == 'X':
                return 10
            elif board[row][0] == 'O':
                return -10

    # Check columns for victory
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            if board[0][col] == 'X':
                return 10
            elif board[0][col] == 'O':
                return -10

    # Check diagonals for victory
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == 'X':
            return 10
        elif board[0][0] == 'O':
            return -10

    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == 'X':
            return 10
        elif board[0][2] == 'O':
            return -10

    # No winner yet
    return 0

# Minimax algorithm
def minimax(board, depth, is_maximizing):
    score = evaluate(board)

    # If X has won, return the score
    if score == 10:
        return score - depth

    # If O has won, return the score
    if score == -10:
        return score + depth

    # If no more moves and no winner, it's a tie
    if not is_moves_left(board):
        return 0

    # If the maximizer's move (X)
    if is_maximizing:
        best = -math.inf

        for i in range(3):
            for j in range(3):
                # Check if the cell is empty
                if board[i][j] == ' ':
                    # Make the move
                    board[i][j] = 'X'

                    # Call minimax recursively and choose the maximum value
                    best = max(best, minimax(board, depth + 1, False))

                    # Undo the move
                    board[i][j] = ' '

        return best

    # If the minimizer's move (O)
    else:
        best = math.inf

        for i in range(3):
            for j in range(3):
                # Check if the cell is empty
                if board[i][j] == ' ':
                    # Make the move
                    board[i][j] = 'O'

                    # Call minimax recursively and choose the minimum value
                    best = min(best, minimax(board, depth + 1, True))

                    # Undo the move
                    board[i][j] = ' '

        return best

# Find the best move for X
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            # Check if the cell is empty
            if board[i][j] == ' ':
                # Make the move
                board[i][j] = 'X'

                # Compute evaluation function for this move
                move_val = minimax(board, 0, False)

                # Undo the move
                board[i][j] = ' '

                # If the value of the current move is more than the best value, update best
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

# Function to print the board
def print_board(board):
    for row in board:
        print(row[0] + ' | ' + row[1] + ' | ' + row[2])
        print('--+---+--')

# Function to check for a winner or tie
def check_winner(board):
    score = evaluate(board)
    if score == 10:
        return "X wins!"
    elif score == -10:
        return "O wins!"
    elif not is_moves_left(board):
        return "It's a tie!"
    else:
        return None

# Main function to simulate the game
def play_game():
    print("Welcome to Tic-Tac-Toe with Minimax!")
    current_turn = 'X'

    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(winner)
            break

        if current_turn == 'X':
            print("X's Turn:")
            best_move = find_best_move(board)
            board[best_move[0]][best_move[1]] = 'X'
            current_turn = 'O'
        else:
            print("O's Turn:")
            row, col = map(int, input("Enter row and column (0-2): ").split())
            if board[row][col] == ' ':
                board[row][col] = 'O'
                current_turn = 'X'
            else:
                print("Invalid move, try again.")

# Start the game
play_game()


5B. Write a program to shuffle Deck of Cards
# import modules
import itertools, random
# make a deck of cards
deck = list(itertools.product(range(1, 14), ['Spade', 'Heart', 'Diamond', 'Club']))
# shuffle the cards
random.shuffle(deck)
# draw five cards
print("You got:")
for i in range(5):
    print(deck[i][0], "of", deck[i][1])


6A. Design an application to stimulate number puzzle problem
from __future__ import print_function
from simpleai.search import astar, SearchProblem
from simpleai.search.viewers import WebViewer

# The goal state of the puzzle
GOAL = '''1-2-3
4-5-6
7-8-e'''

# The initial state of the puzzle
INITIAL = '''4-1-2
7-e-3
8-5-6'''

# Convert list representation to a string
def list_to_string(list_):
    return '\n'.join(['-'.join(row) for row in list_])

# Convert string representation to a list
def string_to_list(string_):
    return [row.split('-') for row in string_.split('\n')]

# Find the location of a specific element in the puzzle
def find_location(rows, element_to_find):
    '''Find the location of a piece in the puzzle. Returns a tuple: (row, column)'''
    for ir, row in enumerate(rows):
        for ic, element in enumerate(row):
            if element == element_to_find:
                return ir, ic

# Create a cache for the goal position of each piece to avoid recalculating
goal_positions = {}
rows_goal = string_to_list(GOAL)
for number in '12345678e':
    goal_positions[number] = find_location(rows_goal, number)

# Define the puzzle problem class
class EigthPuzzleProblem(SearchProblem):
    def actions(self, state):
        '''Returns a list of the pieces we can move to the empty space.'''
        rows = string_to_list(state)
        row_e, col_e = find_location(rows, 'e')
        actions = []
        if row_e > 0:
            actions.append(rows[row_e - 1][col_e])
        if row_e < 2:
            actions.append(rows[row_e + 1][col_e])
        if col_e > 0:
            actions.append(rows[row_e][col_e - 1])
        if col_e < 2:
            actions.append(rows[row_e][col_e + 1])
        return actions

    def result(self, state, action):
        '''Return the resulting state after moving a piece to the empty space.'''
        rows = string_to_list(state)
        row_e, col_e = find_location(rows, 'e')
        row_n, col_n = find_location(rows, action)
        rows[row_e][col_e], rows[row_n][col_n] = rows[row_n][col_n], rows[row_e][col_e]
        return list_to_string(rows)

    def is_goal(self, state):
        '''Returns True if the state is the goal state.'''
        return state == GOAL

    def cost(self, state1, action, state2):
        '''Returns the cost of performing an action. Not useful in this problem but needed.'''
        return 1

    def heuristic(self, state):
        '''Returns an estimate of the distance from a state to the goal using Manhattan distance.'''
        rows = string_to_list(state)
        distance = 0
        for number in '12345678e':
            row_n, col_n = find_location(rows, number)
            row_n_goal, col_n_goal = goal_positions[number]
            distance += abs(row_n - row_n_goal) + abs(col_n - col_n_goal)
        return distance

# Solve the puzzle using A* algorithm
result = astar(EigthPuzzleProblem(INITIAL))

# Print the solution path
for action, state in result.path():
    print('Move number', action)
    print(state)


7A. Solve constraint satisfaction problem
from simpleai.search import CspProblem, backtrack, min_conflicts
from simpleai.search.viewers import ConsoleViewer
from simpleai.search.csp import (
    MOST_CONSTRAINED_VARIABLE,
    HIGHEST_DEGREE_VARIABLE,
    LEAST_CONSTRAINING_VALUE
)
# Variables: The regions in the map
variables = ('WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T')
# Domain: The possible colors for each region
domains = dict((v, ['red', 'green', 'blue']) for v in variables)
# Constraints function: Neighbors must have different colors
def const_different(variables, values):
    return values[0] != values[1]  # Neighbors must have different values (colors)
# Constraints: Define which regions are adjacent (neighbors)
constraints = [
    (('WA', 'NT'), const_different),
    (('WA', 'SA'), const_different),
    (('NT', 'SA'), const_different),
    (('SA', 'Q'), const_different),
    (('NT', 'Q'), const_different),
    (('SA', 'NSW'), const_different),
    (('Q', 'NSW'), const_different),
    (('SA', 'V'), const_different),
    (('NSW', 'V'), const_different),
]
# Create the CSP problem
my_problem = CspProblem(variables, domains, constraints)
# Solve using different heuristics and print the results
# 1. Default backtracking
print("Solution with default backtracking:")
print(backtrack(my_problem))
# 2. Backtracking with Most Constrained Variable heuristic
print("\nSolution with Most Constrained Variable heuristic:")
print(backtrack(my_problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE))
# 3. Backtracking with Highest Degree Variable heuristic
print("\nSolution with Highest Degree Variable heuristic:")
print(backtrack(my_problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE))
# 4. Backtracking with Least Constraining Value heuristic
print("\nSolution with Least Constraining Value heuristic:")
print(backtrack(my_problem, value_heuristic=LEAST_CONSTRAINING_VALUE))
# 5. Backtracking with Most Constrained Variable and Least Constraining Value heuristic
print("\nSolution with Most Constrained Variable and Least Constraining Value heuristic:")
print(backtrack(my_problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE,
               value_heuristic=LEAST_CONSTRAINING_VALUE))
# 6. Backtracking with Highest Degree Variable and Least Constraining Value heuristic
print("\nSolution with Highest Degree Variable and Least Constraining Value heuristic:")
print(backtrack(my_problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE,
               value_heuristic=LEAST_CONSTRAINING_VALUE))
# 7. Min-conflicts heuristic
print("\nSolution with min-conflicts heuristic:")
print(min_conflicts(my_problem))


8A. Derive the expression based on associative law
def associative_law_and(A, B, C):
    # Associative Law for AND
    left_side = (A and B) and C
    right_side = A and (B and C)
    return left_side == right_side
def associative_law_or(A, B, C):
    # Associative Law for OR
    left_side = (A or B) or C
    right_side = A or (B or C)
    return left_side == right_side
# Test values
A, B, C = True, False, True
# Check Associative Law for AND
and_result = associative_law_and(A, B, C)
print(f"Associative Law for AND: (A AND B) AND C = A AND (B AND C) is {and_result}")
# Check Associative Law for OR
or_result = associative_law_or(A, B, C)
print(f"Associative Law for OR: (A OR B) OR C = A OR (B OR C) is {or_result}")


8(B). Derive the expression based on distributive law
def distributive_law_and_over_or(A, B, C):
    # Distributive Law: A AND (B OR C) = (A AND B) OR (A AND C)
    left_side = A and (B or C)
    right_side = (A and B) or (A and C)
    return left_side == right_side

def distributive_law_or_over_and(A, B, C):
    # Distributive Law: A OR (B AND C) = (A OR B) AND (A OR C)
    left_side = A or (B and C)
    right_side = (A or B) and (A or C)
    return left_side == right_side

# Test values
A, B, C = True, False, True

# Check Distributive Law for AND over OR
and_over_or_result = distributive_law_and_over_or(A, B, C)
print(f"Distributive Law: A AND (B OR C) = (A AND B) OR (A AND C) is {and_over_or_result}")

# Check Distributive Law for OR over AND
or_over_and_result = distributive_law_or_over_and(A, B, C)
print(f"Distributive Law: A OR (B AND C) = (A OR B) AND (A OR C) is {or_over_and_result}")


9A. Define the predicates
# Define the predicates
def is_batsman(person):
    return person == "Sachin"
def is_cricketer(batsman):
    return batsman == "batsman"
# Deriving the conclusion
def derive_cricketer(person):
    if is_batsman(person) and is_cricketer("batsman"):
        return f"{person} is a cricketer"
    return f"{person} is not a cricketer"
# Test the derivation
sachin = "Sachin"
result = derive_cricketer(sachin)
print(result)


10A. Relationships
# Define family members with their gender
male = {
    "John": True,
    "Michael": True,
    "David": True,
    "Robert": True,
    "Steve": True,
    "Daniel": True,
    "Tom": True,
    "Alex": True,
    "Ryan": True,
}
female = {
    "Emily": False,
    "Sarah": False,
    "Jessica": False,     
    "Lisa": False,
    "Anna": False,
    "Karen": False,
    "Nancy": False,
}
# Define parent-child relationships
parents = {
    "John": ["Emily", "Michael"],  # John is the father of Emily and Michael
    "Sarah": ["Emily", "Michael"],  # Sarah is the mother of Emily and Michael
    "Michael": ["David", "Lisa"],   # Michael is the father of David and Lisa
    "Jessica": ["David", "Lisa"],   # Jessica is the mother of David and Lisa
    "Robert": ["Daniel"],             # Robert is the father of Daniel
    "Emily": ["Ryan"],                # Emily is the mother of Ryan
}
# Define rules for family relations
def is_father(name, child):
    return name in male and child in parents.get(name, [])
def is_mother(name, child):
    return name in female and child in parents.get(name, [])
def is_grandfather(grandfather, grandchild):
    return grandfather in male and any(
        is_father(grandfather, parent) and grandchild in parents.get(parent, [])
        for parent in parents
    )
def is_grandmother(grandmother, grandchild):
    return grandmother in female and any(
        is_mother(grandmother, parent) and grandchild in parents.get(parent, [])
        for parent in parents
    )
def is_brother(sibling, person):
    return sibling in male and sibling != person and any(person in parents.get(parent, []) for parent in parents if sibling in parents[parent])
def is_sister(sibling, person):
    return sibling in female and sibling != person and any(person in parents.get(parent, []) for parent in parents if sibling in parents[parent])
def is_uncle(uncle, nephew):
    return uncle in male and any(is_brother(uncle, parent) for parent in parents.get(nephew, []))
def is_aunt(aunt, niece):
    return aunt in female and any(is_sister(aunt, parent) for parent in parents.get(niece, []))
def is_nephew(nephew, uncle_or_aunt):
    return any(is_uncle(uncle_or_aunt, child) for child in parents.get(nephew, []))
def is_cousin(cousin, person):
    return any(is_brother_or_sister(parent, person) for parent in parents.get(cousin, []))
def is_brother_or_sister(sibling, person):
    return sibling != person and any(sibling in parents.get(parent, []) for parent in parents if person in parents[parent])
# Example Queries
print("Is John the father of Emily?", is_father("John", "Emily"))
print("Is Sarah the mother of Michael?", is_mother("Sarah", "Michael"))
print("Is John a grandfather of Ryan?", is_grandfather("John", "Ryan"))
print("Is Sarah a grandmother of David?", is_grandmother("Sarah", "David"))
print("Is Michael a brother of Lisa?", is_brother("Michael", "Lisa"))
print("Is Emily a sister of Michael?", is_sister("Emily", "Michael"))
print("Is John an uncle of Daniel?", is_uncle("John", "Daniel"))
print("Is Jessica an aunt of Ryan?", is_aunt("Jessica", "Ryan"))
print("Is David a cousin of Ryan?", is_cousin("David", "Ryan"))

