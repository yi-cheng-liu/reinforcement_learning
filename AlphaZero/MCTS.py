import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Othello.Arena import Arena
from tqdm import tqdm
from random import shuffle

class PolicyNet(nn.Module):
    def __init__(self, game):
        super().__init__()
        
        # parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_channels = 256  # number of channels for the Conv2d layer
        self.dropout = 0.3  # Dropout probability
        
        # convolutional layers
        self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-2)*(self.board_y-2), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, self.action_size)

        self.fc3 = nn.Linear(512, 1)

    def forward(self, s):
        """
        Args:
            s: board configurtion, torch.Tensor with shape (batch_size, board_x, board_y)
        Returns:
            pi: log probability of actions in state s, torch.Tensor with shape (batch_size, action_size)
            v: value of state s, torch.Tensor with shape (batch_size, 1)
        """
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = s.view(-1, self.num_channels*(self.board_x-2)*(self.board_y-2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 512

        # log probability of actions in state s
        pi = F.log_softmax(self.fc2(s), dim=1)                                                   # batch_size x action_size
        # value of state s
        v = torch.tanh(self.fc3(s))                                                              # batch_size x 1

        return pi, v


class MCTS:
    """
    This class handles the MCTS tree.
    """
    def __init__(self, game, policy_net):
        self.game = game
        self.policy_net = policy_net
        
        self.num_MCTS_sims = 50  # number of simulations for MCTS for each action
        self.bonus_term_factor = 1.0
        
        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores number of times edge s,a was visited
        self.Ns = {}  # stores number of times board s was visited
        self.Ps = {}  # stores initial policy (returned by policy network)

        self.Es = {}  # stores game.getGameEnded for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard):
        """
        This function performs num_MCTS_sims simulations of MCTS starting from
        canonicalBoard.
        
        Args:
            canonicalBoard: canonical board configuration, a 2D numpy array:
                            1=current player, -1=the opponent, 0=empty
                            first dim is row , second is column
        Returns:
            probs: a list with len=action_size, which is a policy vector 
                   where the probability of the ith action is proportional to Nsa[(s,a)]
        """
        # Doing self.num_MCTS_sims times of simulations starting from the state 'canonicalBoard'
        for i in range(self.num_MCTS_sims):
            self.search(canonicalBoard)

        # Use string representation for the state
        s = self.game.stringRepresentation(canonicalBoard)
        """
        Please complete the codes for calculating the updated policy vector 'probs' using 'self.Nsa'
        Some information you may need:
            self.Nsa[(s, a)] stores number of times edge s,a was visited.
            If (s,a) is not in self.Nsa, then s has not been visited.
            self.game.getActionSize() returns the number of actions, i.e., n*n+1.
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one simulation of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        
        This is a recursive function.
        
        Args:
            canonicalBoard: canonical board configuration, a 2D numpy array:
                            1=current player, -1=the opponent, 0=empty
                            first dim is row , second is column
        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        
        # Use string representation for the state
        s = self.game.stringRepresentation(canonicalBoard)
        
        # Update self.Es
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        
        
        if self.Es[s] != 0:  # The game ended, which means that s is a terminal node
            # If the current player won, then return -1 (The value for the other player).
            # Otherwise, return 1 (The value for the other player).
            return -self.Es[s]

        if s not in self.Ps:  # There is no policy for the current state s, which means that s is a leaf node (a new state)
            
            # Set Q(s,a)=0 and N(s,a)=0 for all a
            for a in range(self.game.getActionSize()):
                self.Qsa[(s, a)] = 0
                self.Nsa[(s, a)] = 0
            
            # Calculate the output of the policy network, which are the policy and the value for state s
            board = torch.FloatTensor(canonicalBoard.astype(np.float64)).view(1, self.policy_net.board_x,
                                                                              self.policy_net.board_y)
            self.policy_net.eval()
            with torch.no_grad():
                pi, v = self.policy_net(board)
            self.Ps[s] = torch.exp(pi).data.cpu().numpy()[0]  # The policy for state s
            v = v.data.cpu().numpy()[0][0]  # The value of state s
            
            # Masking invalid moves
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids  # Stores the valid moves
            self.Ns[s] = 0
            return -v
        
        # pick the action with the highest upper confidence bound (ucb) and assign it to best_act
        best_act = -1
        valids = self.Vs[s]
        cur_best = -float('inf')
        for a in range(self.game.getActionSize()):
            if valids[a]:
                """
                Please complete the codes for picking the action with the highest UCB
                Some information you may need:
                    self.Qsa[(s, a)] stores the Q value for s,a
                    self.bonus_term_factor=1.0 is the factor "h" in the UCB (See Eq.(1) in the reference guide)
                    self.Ps stores the policy returned by policy network
                    self.Ps[s][a] is the probability corresponding to state s and action a
                    self.Ns[s] stores the number of times board s was visited
                    self.Nsa[(s, a)] stores number of times edge s,a was visited
                """
                ### BEGIN SOLUTION
                # YOUR CODE HERE
                raise NotImplementedError()
                ### END SOLUTION
        
        # Continue the simulation: take action best_act in the simulation
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)  # This returns the value for the current player
        
        """
        Please complete the codes for updating the Q function ('self.Qsa')
        and the number of times that (s,a) has been visited ('self.Nsa')
        Some information you may need:
            self.Qsa[(s, a)] stores the Q value for s,a
            self.Ns[s] stores the number of times board s was visited
            self.Nsa[(s, a)] stores number of times edge s,a was visited
            v is the value for the current player
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        raise NotImplementedError()
        ### END SOLUTION
        
        # Update the number of times that s has been visited
        self.Ns[s] += 1
        
        return -v
    
    
class Coach():
    """
    This class executes the self-play + learning.
    """
    def __init__(self, game):
        self.game = game
        self.nnet = PolicyNet(game)
        self.pnet = PolicyNet(game)  # the competitor network
        self.mcts = MCTS(game, self.nnet)
        self.epochs = 10  # number of training epochs for each iteration
        self.learning_rate = 0.001
        self.batch_size = 64  # batch size
        self.trainExamples = []  # historical examples for training
        self.numIters = 2  # number of iterations
        self.numEps = 20  # number of complete self-play games for one iteration.
        self.arenaCompare = 40  # number of games to play during arena play to determine if new net will be accepted.
        self.updateThreshold = 0.6  # During arena playoff, new neural net will be accepted if threshold or more of games are won.

    def train(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.numIters + 1):
            print(f'Starting Iter #{i} ...')

            for _ in tqdm(range(self.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet)  # reset search tree
                self.trainExamples.extend(self.executeEpisode()) # save the iteration examples to the history
            
            # shuffle examples before training           
            shuffle(self.trainExamples)

            # training new network, keeping a copy of the old one
            self.pnet.load_state_dict(self.nnet.state_dict())

            optimizer = optim.Adam(self.nnet.parameters(), lr=self.learning_rate)

            for epoch in range(self.epochs):
                print('EPOCH ::: ' + str(epoch + 1))
                self.nnet.train()
                
                """
                Please complete the training codes for self.nnet
                Some information you may need:
                    self.trainExamples is a list that stores historical examples for training
                    self.trainExamples[i] has the form (canonicalBoard, pi, v)
                    The output of self.nnet include pi and v, where
                        pi are the log probabilities of actions in state s;
                        v is the value of state s.
                """
                ### BEGIN SOLUTION
                # YOUR CODE HERE
                raise NotImplementedError()
                ### END SOLUTION
            
            pmcts = MCTS(self.game, self.pnet)
            nmcts = MCTS(self.game, self.nnet)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x)),
                          lambda x: np.argmax(nmcts.getActionProb(x)), self.game)
            pwins, nwins, draws = arena.playGames(self.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_state_dict(self.pnet.state_dict())
            else:
                print('ACCEPTING NEW MODEL')
                self.pnet.load_state_dict(self.nnet.state_dict())
                self.trainExamples = []
        
    def play(self, canonicalBoard):
        """
        Args:
            canonicalBoard: canonical board configuration, a 2D numpy array:
                            1=current player, -1=the opponent, 0=empty
                            first dim is row , second is column
        Returns:
            action: Putting a disc on row x and column y of the board corresponds to action=x*n+y. action=n*n means passing.
            (Row and column are counting from 0 to n-1.) 
        """
        mcts = MCTS(self.game, self.nnet)
        action = np.argmax(mcts.getActionProb(canonicalBoard))
        return action
    
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1 (Black player).
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi, v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, -1 if the player lost the game, and otherwise 0.000001
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            
            # After 10 steps, we use the greedy action rather than a random action
            if episodeStep < 10:
                pi = self.mcts.getActionProb(canonicalBoard)
            else:
                pi = list(np.zeros((self.game.getActionSize(),)))
                pi[np.argmax(self.mcts.getActionProb(canonicalBoard))] = 1
            
            # Add symmetric samples
            sym = self.game.getSymmetries(canonicalBoard, pi)
            
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            
            # Take action according to the policy pi
            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:  # if the current episode of game ended
                trainExamples = [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
                return trainExamples
