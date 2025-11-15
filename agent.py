"""
store all the agents here
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import time
import pickle
from collections import deque
import json

#new imports
import torch
import torch.nn as nn
import numpy as np
import random



# built-in PT loss, a shorter option, but the original code had its own function
#delta=1.0 matches TF code
#huber = nn.SmoothL1Loss(beta=1.0, reduction="mean")  # beta is the delta
#loss = huber(y_pred, y_true)

def huber_loss_torch(y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     delta: float = 1.0,
                     reduction: str = "mean") -> torch.Tensor:
    """
    PyTorch equivalent of Keras huber_loss/mean_huber_loss.
    reduction: "none" | "mean" | "sum"
    """
    error = y_true - y_pred #error difference
    abs_err = torch.abs(error) #absolute value
    quad = 0.5 * error.pow(2) #calc for quadratic
    lin  = delta * (abs_err - 0.5 * delta) #calc for linear
    out = torch.where(abs_err < delta, quad, lin) #if error < delta use quad, else use mean
    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out

def mean_huber_loss_torch(y_true, y_pred, delta: float = 1.0) -> torch.Tensor:
    return huber_loss_torch(y_true, y_pred, delta=delta, reduction="mean")


#no TF - stays as is:
class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col

#PyTprch uses Modules (unlike TF, calls layers as functions on tensors directly)
#so we need a class for the model that includes foreward function, unpacks JSON file, handles permute and padding from keras code osv
class TorchDQN(nn.Module):
    """
    Builds a Keras-like DQN from a JSON config and produces Q-values (B, n_actions).
    Accepts inputs as (B, H, W, C) and handles the NHWC->NCHW permute internally.
    """
    def __init__(self, board_size: int, n_frames: int, n_actions: int, cfg: dict):
        super().__init__()
        self.board_size = board_size 
        self.n_frames = n_frames #channels
        self.n_actions = n_actions #
        self.cfg= cfg #json config

        layers = []
        in_c = n_frames  # channels = frames for Conv2d

        #parse the json file and find keys like "Conv2D_1", "Flatten_1", "Dense_1".
        for key, spec in self.cfg["model"].items():
            #Keras: 'same'/'valid' -> PyTorch: 'same'/0
            padding_cfg = spec.get("padding", "valid")
            padding = "same" if padding_cfg == "same" else 0
            if "Conv2D" in key:
                out_c    = int(spec.get("filters"))
                ksize    = spec.get("kernel_size", 3)
                stride   = spec.get("strides", 1)
                padding  = padding
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=ksize, stride=stride, padding=padding, bias=True))
                layers.append(self._act(spec.get("activation", "linear")))
                in_c = out_c

            elif "MaxPool2D" in key or "MaxPooling2D" in key:
                pool     = spec.get("pool_size", 2)
                stride   = spec.get("strides", pool)
                layers.append(nn.MaxPool2d(kernel_size=pool, stride=stride))

            elif "Flatten" in key: #flattens everything for libnear layers
                layers.append(nn.Flatten())

            elif "Dense" in key: #linear in PyTorch
                units = int(spec.get("units"))
                # LazyLinear lets PyTorch infer in_features on first forward
                layers.append(nn.LazyLinear(units)) #lazy to not compute in_features by hand
                layers.append(self._act(spec.get("activation", "linear")))

            else:
                # Unknown layer type: skip or print a warning
                print(f"Warning: unrecognized layer {key}, skipping.")

        # Final linear head to n_actions (matches Keras 'action_values' layer with linear activation)
        layers.append(nn.LazyLinear(self.n_actions))

        self.net = nn.Sequential(*layers) 

    #helpers
    #activation functions from Keras need to be mapped to PyTorch Module
    def _act(self,name: str) -> nn.Module:
        name = (name or "linear").lower()
        if name == "relu":return nn.ReLU()
        if name == "tanh": return nn.Tanh()
        if name == "sigmoid": return nn.Sigmoid()
        if name in ("leakyrelu", "leaky_relu"): return nn.LeakyReLU()
        return nn.Identity()  #'linear' / default

    #forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as (B, H, W, C); Conv2d expects (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous() #cange order
        z = self.net(x)  #may already be flat if Flatten/Dense are in JSON
        if z.ndim > 2:   # safety: flatten if JSON forgot a Flatten before a Dense
            z = torch.flatten(z, 1)
        return z  # (B, n_actions)

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values

    Attributes
    ----------
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _target_net : TensorFlow Graph
        Stores the target network graph of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.device = torch.device("cpu") 
        self.reset_models()

    def reset_models(self):
        """create nn.Module, copy state dicts, set optimizer and loss"""
        self.q = self._agent_model().to(self.device) #call _agent_model() to create an instance of the model
        if self._use_target_net: #if using target network, create a clone of the online network:
            self.q_target = self._agent_model().to(self.device)
            self.q_target.load_state_dict(self.q.state_dict())
            self.q_target.eval()
        self.optim = torch.optim.RMSprop(self.q.parameters(), lr=5e-4)  #optimizer, same as Keras
        self.criterion = lambda y_pred, y_true: mean_huber_loss_torch(y_true, y_pred, delta=1.0) #loss function

    def _prepare_input(self, board)-> torch.Tensor:
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        t = torch.from_numpy(board).to(self.device).float() #return a torch tensor instead of just board
        return t

    def _get_model_outputs(self, board, model=None)-> np.ndarray:
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : model

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        if model is None:
            model = self.q
        x = self._prepare_input(board)#torch tensor
        model.eval() #evaluation mode
        with torch.no_grad(): #run forward pass without gradients
            q = model(x)#tensor for Q vales, shape (B, n_actions) 
        #return: remove gradients if not done, move from cpu, convert to numpy array:    
        return q.detach().cpu().numpy() 

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.astype(np.float32)/4.0 #no change

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        q = self._get_model_outputs(board, self.q)
        masked = np.where(legal_moves == 1, q, -np.inf) #ignore illegal moves (if legal keep Q value, if illegal set to -inf)
        return np.argmax(masked, axis=1).astype(np.int64) #choose max Q and convert to 64 bit integer

    def _agent_model(self): 
        """creates an instance of TorchDQN class"""
        with open(f"model_config/{self._version}.json", "r") as f:
            cfg = json.load(f)
        return TorchDQN(self._board_size, self._n_frames, self._n_actions, cfg)

    def get_action_proba(self, board):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        q = self._get_model_outputs(board, self.q)#Q values for the model for each action
        q = np.clip(q, -10, 10) #keep values within the range [-10,10]
        q = q - q.max(axis=1, keepdims=True) #subtract max from each value in the row to prevent exp overflow
        probs = np.exp(q) #e^Q raw Q->possitive values
        probs /= probs.sum(axis=1, keepdims=True) #softmax: normalize probs so the row sums to 1
        return probs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk with torch.save in pt format
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        iter = 0 if iteration is None else int(iteration)
        torch.save(self.q.state_dict(), f"{file_path}/model_{iter:04d}.pt")
        if self._use_target_net:
            torch.save(self.q_target.state_dict(), f"{file_path}/model_{iter:04d}_target.pt")


    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available 
        Load models from disk using torch.load function
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        iter = 0 if iteration is None else int(iteration) #itreation number
        self.q.load_state_dict(torch.load(f"{file_path}/model_{iter:04d}.pt", map_location=self.device)) #load file and convert it back to tensor, load to device
        if self._use_target_net: #load target net if exists
            self.q_target.load_state_dict(torch.load(f"{file_path}/model_{iter:04d}_target.pt", map_location=self.device))

    def print_models(self):
        """Print the current models"""
        print(self.q)
        if self._use_target_net: print(self.q_target)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        1.sample minibatch from replay buffer
        2.convert values s, a, r, next_s, done, legal_moves to tensors and normalize
        3.normalize action format (one hot/integer)
        4. Reward clipping if activated
        5. compute TD target t without grads with Bellman target: y = r + γ * max_next * (1 − done)
        6. predict current Q(s,_) and select Q(s,a)
        7. calculate loss (Huber)
        8. backpropagation, optimization
        9.return loss (scalar)

        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : scalar
        """
        #1.Sample a minibatch (NumPy arrays)
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        # 2.Convert to torch tensors on the right device (+ normalize states)
        device = self.device
        s = torch.as_tensor(self._normalize_board(s), dtype=torch.float32, device=device)#(B,H,W,F)
        next_s = torch.as_tensor(self._normalize_board(next_s), dtype=torch.float32, device=device)#(B,H,W,F)
        r= torch.as_tensor(r, dtype=torch.float32, device=device) #(B,1)
        done = torch.as_tensor(done, dtype=torch.float32, device=device)  #(B,1)
        #legal_moves is 0/1: 
        legal_moves = torch.as_tensor(legal_moves, dtype=torch.bool, device=device)  #(B,A)

        # actions: accept one-hot or integer indices (keras handled automatically, pytorch can't)
        if isinstance(a, np.ndarray):
            a = torch.as_tensor(a, device=device) #make sure a is converted to tensor
        if a.ndim == 2 and a.shape[1] > 1: #if its one hot - convert to (B,1)
            a_idx = a.argmax(dim=1, keepdim=True)
        else:                                            
            a_idx = a.long().view(-1, 1) #if an integer check if shape is (B,1)

        if reward_clip:
            r = r.sign() #clip for stability: {-1,0,1}

        self.q.train() #€activate training mode

        # 3.Compute TD target y = r + γ * max_a' Q_target(next_s, a') * (1 - done)
        with torch.no_grad(): #no grads in target network
            q_next = (self.q_target if self._use_target_net else self.q)(next_s) #q-values for next state
            q_next = q_next.masked_fill(~legal_moves, float('-inf')) # mask illegal actions → set to -inf so they never win the max
            max_next = q_next.max(dim=1, keepdim=True).values #pick the best action for next state
            y = r + self._gamma * max_next * (1.0 - done) 

        #4.Q(s, a) for the taken actions
        q_s  = self.q(s) #current q values
        q_sa = q_s.gather(1, a_idx) # predicted values for chosen action

        #5. Huber loss on the chosen action
        loss = mean_huber_loss_torch(y_true=y, y_pred=q_sa, delta=1.0)

        # 6.Backprop + step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.item())

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently.
        Copy weights from the main Q-network to the target network."""
        if self._use_target_net:
            self.q_target.load_state_dict(self.q.state_dict())

    