##############
from collections import defaultdict
import numpy as np
import random
import copy
import gzip
import pickle

class Node:

  def __init__(self, state, wins, parent, action, playouts, player):

    self.state = state
    self.wins = wins
    self.parent = parent
    self.action = action
    self.playouts = playouts
    self.children = []
    self.player = player



# function to make the move in column 'col' with player 'player' and returns whether it could move and the final board
def makeMove(board, col, player, rows):

  isSuccess = False

  for i in range(rows-1,-1,-1):

    if board[i][col] == 0:
       board[i][col] = player
       isSuccess = True
       break 

  return board,isSuccess 



# returns true if a move can be made else false
def isMovePossible(board, rows):

  possible = False

  for i in range(0,5):

    for j in range(0,rows):

      if board[j][i] == 0:
        possible = True
        break
  
  return possible




# returns the next player to play
def nextPlayer(player):

  if player == 1:
    return 2

  else:
    return 1


# returns true if a move can be made from and after column 'col' else false
def isMovePossibleFromCol(board, col, rows):

  i = col
  # movePossible = False

  while i<5:

    for j in range(rows):
      
      if board[j][i] == 0:
        return True
      
    i += 1
  
  return False



# takes the board as input and returns whether game has ended and the result, result is 1 if P1 wins, 2 if P2 wins and 0 if draw.

def gameResult2(board, rows):

  result = 0

# check in rows

  for i in range(0,rows):

    if (board[i][0] == 1 and board[i][1] == 1 and board[i][2] == 1 and board[i][3] == 1) or (board[i][1] == 1 and board[i][2] == 1 and board[i][3] == 1 and board[i][4] == 1):
      result = 1
      break

# check in columns
  for i in range(0,5):

     j = 0
     while j+3 < rows:
      if (board[j][i] == 1 and board[j+1][i] == 1 and board[j+2][i] == 1 and board[j+3][i] == 1):
        result = 1
        break
    
      j += 1

  
  # DIAGNOLS

  # left to right

  for r in range(rows):
    for c in range(5):

      if r+3 < rows and c+3 < 5:
        
        if (board[r][c] == 1 and board[r+1][c+1] == 1 and board[r+2][c+2] == 1 and board[r+3][c+3] == 1):
          result = 1
          break

    if result == 1:
      break


  # right to left

  for r in range(rows):
    for c in range(5):

      if r+3 < rows and c-3 >= 0:
        
        if (board[r][c] == 1 and board[r+1][c-1] == 1 and board[r+2][c-2] == 1 and board[r+3][c-3] == 1):
          result = 1
          break

    if result == 1:
      break
  
  
  if result == 1:
    return True,1


## CHECK FOR Player 2 NOW

  # check in rows

  for i in range(0,rows):

    if (board[i][0] == 2 and board[i][1] == 2 and board[i][2] == 2 and board[i][3] == 2) or (board[i][1] == 2 and board[i][2] == 2 and board[i][3] == 2 and board[i][4] == 2):
      result = 2
      break

# check in columns
  for i in range(0,5):

     j = 0
     while j+3 < rows:
      if (board[j][i] == 2 and board[j+1][i] == 2 and board[j+2][i] == 2 and board[j+3][i] == 2):
        result = 2
        break
    
      j += 1

  
  # DIAGNOLS

  # left to right

  for r in range(rows):
    for c in range(5):

      if r+3 < rows and c+3 < 5:
        
        if (board[r][c] == 2 and board[r+1][c+1] == 2 and board[r+2][c+2] == 2 and board[r+3][c+3] == 2):
          result = 2
          break

    if result == 2:
      break


  # right to left

  for r in range(rows):
    for c in range(5):

      if r+3 < rows and c-3 >= 0:
        
        if (board[r][c] == 2 and board[r+1][c-1] == 2 and board[r+2][c-2] == 2 and board[r+3][c-3] == 2):
          result = 2
          break

    if result == 2:
      break
    
  if result == 2:
    return True,2

  
  if isMovePossible(board, rows) == True:
    return False, -1
  
  else:
    return True, 0
  
# returns the index of the child with maximum UCB Value

def UCB(n):

  c = 1.414
  return (n.wins/n.playouts) + np.sqrt(c*c*np.log(n.parent.playouts)/n.playouts)


def getUCB(n):

  values = []
  l = len(n.children)
  for i in range(l):

    v = UCB(n.children[i])
    values.append((v,i))

  return values

def getMaximum(values):

  max = 0
  maxIdx = -1

  for i in range(5):

    if values[i][0] > max:

      max = values[i][0]
      maxIdx = values[i][1]
  
  return maxIdx    


  # to select the leaf of the tree
def select(n):

  root = n
  isLeaf = False

  while isLeaf == False:

    if len(root.children) == 5:

      values = getUCB(root)
      idx = getMaximum(values) 
      root = root.children[idx]
    
    else:
      isLeaf = True

  return root


  # make a child from leaf 
def expand(n, rows):

  root = n
  isSuccess = False
  move = len(root.children) + 1
  board = copy.deepcopy(n.state)
  player = 1 if n.player == 1 else 2
  child = None 

  while isSuccess == False and move <=5 :
      # print("greater than 4")
    
    M = makeMove(board, move-1, nextPlayer(player), rows)
    moveMade = M[1]
    board = M[0]

    if moveMade == True:

      isSuccess = True
      child = Node(board, 0, root, move, 0, nextPlayer(player))
      root.children.append(child)

    else:
      move += 1

  return child


  # returns 1 if the current player wins 0 if other player wins and -1 for a draw
def simulate(n, rows):

  board = copy.deepcopy(n.state)
  Originalplayer = 1 if n.player == 1 else 2
  player = Originalplayer
  gameTerminate = False


  while gameTerminate == False:

    R = gameResult2(board,rows)
    ended = R[0]
    winner = R[1]

    # if the game has ended
    if ended == True:

      gameTerminate = True
      if winner == 0:

        # print("game drawn")  
        result = -1
      
      elif winner == 1:

        # print("Player 1 won")

        if Originalplayer == 1:
          result = 1
        else:
          result = 0
      
      else:

        # print("Player 2 won")

        if Originalplayer == 2:
          result = 1
        else:
          result = 0

   #if no-one has won/draw
    else :

      moveSuccess = False
      while moveSuccess == False:

        idx = random.randint(0,4)
        M = makeMove(board, idx, player, rows)

        if M[1] == True:

          moveSuccess = True
          board = M[0]
          break

    player = nextPlayer(player)

  return result


  # function to backpropogate and update the wins and playouts
# Rewards are as follows: 
# win : + 1
# lose: -1
# draw: 0
def backPropogate(tree, result, child):

  #current player wins
  if result == 1:

    child.wins += 1
    child.playouts += 1
    parent = child.parent
    count = 0

    while parent != None:

      if count%2 == 1:
        parent.wins += 1
      else:
        parent.wins -= 1

      parent.playouts += 1 
      parent = parent.parent
      count += 1
  
  #other player wins
  elif result == 0:

    child.wins -= 1
    child.playouts += 1
    parent = child.parent
    count = 0

    while parent != None:

      if count%2 == 0:
        parent.wins += 1
      else:
        parent.wins -= 1

      parent.playouts += 1 
      parent = parent.parent
      count += 1

  #draw
  else:

    child.playouts += 1
    parent = child.parent

    while parent != None:

      parent.playouts += 1 
      parent = parent.parent

  
  return tree


# returns the action with most playouts
def mostPlayouts(n):

  result = None
  max = -1
  l = len(n.children)

  for i in range (0, l):

    if n.children[i].playouts > max :

      max = n.children[i].playouts
      result = n.children[i]

  action = result.action

  return action


  # returns the best move according to the state, number of simulation from the persepective of player 'p' in a game of 'rows' rows*5
def MCTS(state, simulations, p, rows):

  if simulations == 0:
    return random.randint(1,5)

  player = p
  tree = Node(state, 0, None, 0, 0, player)
  # print("Tree node made \n")

  
  for i in range(simulations):
 
    leaf = select(tree)
    
    if len(leaf.children) >= 5:
      print("leaf is incorrect")


    # check if the leaf is a draw state, if yes then only backpropogate else do everything
    col = len(leaf.children)  
    if isMovePossibleFromCol(leaf.state, col, rows) == True:


      child = expand(leaf, rows)      
      # print(leaf.state)
      # print(child.state)
      # print("made a child \n")
      result = simulate(child, rows)
      # print("simulation successful \n")
      tree = backPropogate(tree, result, child)
      # print("back propogation done \n")

    else:

      backPropogate(tree,-1,leaf)

    player = nextPlayer(player)

  # return the move of the node root having the highest number of playouts 
  bestAction = mostPlayouts(tree)
  return  bestAction


  # function to simulate the game between 2 player 'games' number of times with 'rows' number of rows  
def ConnectFour_parta(games, rows):

  mcts200 = 0
  mcts40 = 0

  games = int(games/2)

  for i in range(games):

    board = np.zeros((6,5))

    gameOver = False
    moveCount = 0

    while gameOver == False:

      # print(board)
      M = gameResult2(board,rows)
      result = M[0]
      winner = M[1]

      if result == True:

        gameOver = True
        
        if winner == 1:

          print("MCTS 40 wins")
          mcts40 += 1
        
        elif winner == 2:

          print("MCTS 200 wins")
          mcts200 += 1
        
        else:

          print("Game Draw")

      else:

        if moveCount%2 == 0:

          # print("Player 1 has to move")
          col1 = MCTS(board, 40, 1, rows)

          if(col1 > 5):
            print("col greater than 5")

          T1 = makeMove(board, col1-1, 1, rows)
          if T1[1] == True:
            board = T1[0]
          
        else:

          # print("Player 2 has to move")

          # player 2 is MCTS(200) 
          col2 = MCTS(board, 200, 2, rows)
          
          if(col2 > 5):
            print("col greater than 5")

          T2 = makeMove(board, col2-1, 2, rows)
          if T2[1] == True:
            board = T2[0]
        
        moveCount += 1

  for i in range(games):

    board = np.zeros((6,5))

    gameOver = False
    moveCount = 0

    while gameOver == False:

      # print(board)
      M = gameResult2(board,rows)
      result = M[0]
      winner = M[1]

      if result == True:

        gameOver = True
        
        if winner == 1:

          print("MCTS 200 wins")
          mcts200 += 1
        
        elif winner == 2:

          print("MCTS 40 wins")
          mcts40 += 1
        
        else:

          print("Game Draw")

      else:

        if moveCount%2 == 0:

          # print("Player 1 has to move")
          col1 = MCTS(board, 200, 1, rows)

          if(col1 > 5):
            print("col greater than 5")
          T1 = makeMove(board, col1-1, 1, rows)
          if T1[1] == True:
            board = T1[0]
          
        else:

          # print("Player 2 has to move")

          # If player 2 is MCTS(200)/MCTS(40) 
          col2 = MCTS(board, 40, 2, rows)
          
          if(col2 > 5):
            print("col greater than 5")

          T2 = makeMove(board, col2-1, 2, rows)
          if T2[1] == True:
            board = T2[0]
        
        moveCount += 1

  draw = games + games - mcts200 
  draw = draw - mcts40  
  print("MCTS 200 wins", mcts200 , "times \n")
  print("MCTS 40 wins", mcts40 , "times \n")
  print("Game Drawn", draw , "times \n")


# function to simulate the game between 2 player 'games' number of times with 'rows' number of rows  
def ConnectFour_partc(games, rows, stateActionsMap):

  mcts25 = 0
  qlearning = 0

  for i in range(games):

    board = np.zeros((rows,5))

    gameOver = False
    moveCount = 0

    while gameOver == False:

      M = gameResult2(board,rows)
      result = M[0]
      winner = M[1]

      if result == True:

        gameOver = True
        
        if winner == 1:

          print("MCTS 5 wins")
          mcts25 += 1
        
        elif winner == 2:

          print("Q-Learning Wins")
          qlearning += 1
        
        else:

          print("Game Draw")

      else:

        if moveCount%2 == 0:

          # print("Player 1 has to move")
          col1 = MCTS(board, 25, 1, rows)
          if(col1 > 5):
            print("col greater than 5")

          T1 = makeMove(board, col1-1, 1, rows)
          if T1[1] == True:
            board = T1[0]
          
        else:
          
          # If player 2 is Q-Learning
          col2 = Qlearning_move(board, stateActionsMap)

          if(col2 > 5):
            print("col greater than 5")

          T2 = makeMove(board, col2-1, 2, rows)
          if T2[1] == True:
            board = T2[0]
        
        moveCount += 1

  draw = games - qlearning 
  draw = draw - mcts25  
  print("MCTS 25 wins", mcts25 , "times \n")
  print("Q-Learning wins", qlearning , "times \n")
  print("Game Drawn", draw , "times \n")


# updates the Q(S,A) value for the node 
def update(key, state, reward, alpha, stateActionsMap):
   
  # alpha = 0.8
  gamma = 0.5
  max = -10000000

  if key in stateActionsMap:
    v = stateActionsMap[key]
  else:
    v = 0
    print("not in map !!!!!")
    print(key)
  
  for i in range(1,6):

    newKey = (state,i)
    if newKey not in stateActionsMap:
      val = 0
    else:
      val = stateActionsMap[newKey]

    if val > max:
      max = val

  return (v + alpha*(reward + gamma*max - v))


def epsilon_greedy(values,epsilon):

  action = 0
  max = values[0][0] - 1

  # epsilon = 0.0
  rand = random.uniform(0, 1)

  if rand < epsilon: 
    #select random
    r = random.choices(values)
    action = r[0][1]


  else:  
    for i in range(len(values)):

      if values[i][0] > max:
        max = values[i][0]
        action = values[i][1]
    
  
  return action


def convertTotuple(ini_array):

    # convert numpy arrays into tuples
  result = tuple([tuple(row) for row in ini_array])
  # print result
  return result


  # Train Q-Learning Algorithm for 'episodes' number of trials and a board with 'r' rows
def TrainQlearning(episodes, r):

  stateActionsMap = defaultdict(float)
  epsilon = 1
  alpha = 1
  for i in range(episodes):
    
    player = 1
    terminalState = False
    board = np.zeros((r,5))

    if (i+1)%5000 == 0:
      print("episode: ", i+1)
      print("states explored =",len(stateActionsMap)) 
      print("new epsilon = ", epsilon)
      print("new alpha = ", alpha)
      print("--------------------------------------")

    if (i+1)%10000 == 0:
      epsilon = epsilon*0.8
      alpha = alpha*0.9 

    while terminalState == False:
      
      # print(board)

      if player == 1:

        move_p1 = MCTS(board, 30, player,r)
        M = makeMove(board, move_p1 - 1, player, r)
        board = copy.deepcopy(M[0])
        currentState = copy.deepcopy(board)
        player = nextPlayer(player)
        result = gameResult2(currentState, r)

        if result[0] == True:
          terminalState = True

      # player 2  
      else:

        # see the possible actions 
        possible_actions = []

        for action in [1,2,3,4,5]:
          
          col = action - 1
          if currentState[0][col] == 0:
            possible_actions.append(action)


        prev_state = copy.deepcopy(board)
        convertedState = convertTotuple(currentState)
        action_values = []

        # print(possible_actions)

        for action in possible_actions:

          # print(action, end=" ")
          
          key = (convertedState,action)

          if key not in stateActionsMap:
            stateActionsMap[key] = 0.0
          
          action_values.append((stateActionsMap[key],action))

        
        # for i in range(len(action_values)):
        #   print(action_values[i][0], end=" ")

        move_p2 = epsilon_greedy(action_values, epsilon) # move according to epsilon greedy

        if move_p2 == 0:
          print("wrong move")

        convertedPrev = convertTotuple(prev_state) #convert to tuple
        N = makeMove(board, move_p2 - 1, player, r) # make move
        board = copy.deepcopy(N[0])
        currentState = copy.deepcopy(board)
        convertedState = convertTotuple(currentState)
        player = nextPlayer(player)
        result = gameResult2(currentState, r)
        reward = -1

        if result[0] == True:
          terminalState = True
          
          if result[1] == 2:
            reward = 5
          
          elif result[1] == 1:
            reward = -5
          
          else:
            reward = 0

        action = move_p2 # action taken
        key = (convertedPrev, action)

        stateActionsMap[key] = update(key, convertedState, reward, alpha, stateActionsMap)

  
  
# returns the best move corresponding to a state by Q-Learning

def value_state(state, stateActionsMap):
    stateCopy = copy.deepcopy(state)
    convertedState = convertTotuple(stateCopy)
    max = np.NINF

    for action in [1,2,3,4,5]:
        key = (convertedState, action)
        val = stateActionsMap[key]

        if val > max:

            select = action
            max = val
        
    return max
        


def Qlearning_move(state, stateActionsMap):

  stateCopy = copy.deepcopy(state)
  convertedState = convertTotuple(stateCopy)
  max = -150000

  for action in [1,2,3,4,5]:

    key = (convertedState, action)

    if key in stateActionsMap:
      val = stateActionsMap[key]

      if val > max:

        select = action
        max = val

    else:
      select = random.randint(1,5)
      # print(key)

  return select

def load_model(filename):

    q_val = defaultdict(float)

    try:

        with gzip.open(filename, 'rb') as handle:

            q_val = pickle.load(handle)
    
        print('\nQ_Values Loaded')
        return q_val

    except:
        return q_val


def save_model(filename, stateActionsMap):

  with gzip.open(filename, 'wb') as handle:
            pickle.dump(stateActionsMap, handle, protocol = pickle.HIGHEST_PROTOCOL)







#Your program can go here.

###############

def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()

def main():
    
 user_input = input('\nPlease Enter Your Choice : (Type A for part a output/ Type C for part c output) \n')
 if user_input ==  'A':

    board = np.zeros((6,5))
    rows = 6
    gameOver = False
    moveCount = 0

    while gameOver == False:

      print(board)
      print("\n")
      M = gameResult2(board,rows)
      result = M[0]
      winner = M[1]

      if result == True:

        gameOver = True
        
        if winner == 1:

          print("MCTS 200 wins")
        #   mcts200 += 1
        
        elif winner == 2:

          print("MCTS 40 wins")
        #   mcts40 += 1
        
        else:

          print("Game Draw")

      else:

        if moveCount%2 == 0:

          print("Player 1 MCTS(200) \n")
          col1 = MCTS(board, 200, 1, rows)
          print("Action Selected = ", col1, "\n")
          if(col1 > 5):
            print("col greater than 5")
          

          T1 = makeMove(board, col1-1, 1, rows)
          if T1[1] == True:
            board = T1[0]
          
        else:

          print("Player 2 MCTS(40) \n")

          # If player 2 is MCTS(200)/MCTS(40) 
          col2 = MCTS(board, 40, 2, rows)
          print("Action Selected = ", col2, "\n")
          if(col2 > 5):
            print("col greater than 5")

          T2 = makeMove(board, col2-1, 2, rows)
          if T2[1] == True:
            board = T2[0]
        
        moveCount += 1

    print("Total moves taken = ", moveCount)
        
 else:

    rows = 6 
    board = np.zeros((rows,5))
    gameOver = False
    moveCount = 0
    stateActionsMap = load_model("2018B3A70754G_MANTAVYA.dat.gz")

    while gameOver == False:
    
      print(board)
      print("\n")
      M = gameResult2(board,rows)
      result = M[0]
      winner = M[1]

      if result == True:

        gameOver = True
        
        if winner == 1:

          print("MCTS 25 wins")
        #   mcts25 += 1
        
        elif winner == 2:

          print("Q-Learning Wins")
        #   qlearning += 1
        
        else:

          print("Game Draw")

      else:

        if moveCount%2 == 0:

          # print("Player 1 has to move")
          print("Player 1 MCTS(25) \n")
          col1 = MCTS(board, 25, 1, rows)
          if(col1 > 5):
            print("col greater than 5")

          print("Action Selected = ", col1, "\n")
 
          T1 = makeMove(board, col1-1, 1, rows)
          if T1[1] == True:
            board = T1[0]
          
        else:
          
          # If player 2 is Q-Learning
          print("Player 2 Q-Learning \n")
          col2 = Qlearning_move(board, stateActionsMap)

          if(col2 > 5):
            print("col greater than 5")

          print("Action Selected = ", col2, "\n")

          print("Value of next State ", value_state(board, stateActionsMap), "\n")
          T2 = makeMove(board, col2-1, 2, rows)
          if T2[1] == True:
            board = T2[0]
        
        moveCount += 1

    print("Total moves taken = ", moveCount)


    
if __name__=='__main__':
    main()