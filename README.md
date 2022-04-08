# Connect-4-Game
Playing connect-4 game using Artificial Intelligence. One player is trained using Q-Learning and other using Monte Carlo Tree Search (MCTS).

## Overview
Monte Carlo Tree Search (MCTS) and Q Learning to play the Connect-4 game. 
First we make the MCTS play against itself i.e. MCTS(40) v/s MCTS(200), the number in parenthesis is the number of simulations gone by the algorithm to find the optimal move. 

Then we introduce the Q-Learning algorithm to MCTS. We train the Q-Learning algorithm based on the moves taken by MCTS, and use those trained weights to make it play against MCTS(n), n varies from 0 to 25.


