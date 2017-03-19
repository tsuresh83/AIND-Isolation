"""This file contains all the classes you must complete for this project.
You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.
You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

EPSILON = 1e-5

def custom_score(game,plaer):
    return custom_score_distaceWeightedPositions(game,plaer)

def getEdges (width,height):
    columnEdges = [[r,c] for r in range(height) for c in [0,width-1]]
    rowEdges = [[r,c] for r in [0,height-1] for c in range(width)]
    return columnEdges+rowEdges

def getCorners(width,height):
    return([[0,0],[0,width-1],[height-1,0],[height-1,width-1]])

def calculateDistanceFromCenter(location,boardWidth,boardHeight,invert=False):
    boardCenterCoordinates = [boardWidth/2,boardHeight/2]
    if invert:
        return 1-((abs(boardCenterCoordinates[0]-location[0])+abs(boardCenterCoordinates[1]-location[1]))/sum(boardCenterCoordinates))
    return (abs(boardCenterCoordinates[0]-location[0])+abs(boardCenterCoordinates[1]-location[1]))/sum(boardCenterCoordinates)
def custom_score_distaceWeightedPositions(game,player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    The score is calculated as the ratio between distance weighted sum of 
    player's moves and the distance weighted sum of opponent's moves. The distances
    are calculated from the center of the board. Player's moves that are closer
    to the center have greater weights. Similarly opponents moves have higher weights
    with increasing distance from center. The intuition is to give higher scores
    to player moves that are closer to the center while at the same time pushing opponent
    moves closer to the edges. The whole fraction is normalized by blank spaces
    left on the board (which diminishes with time), so that the same distance weighted
    fraction scores more later in the game than earlier. The EPSILON is to prevent
    division by zero. This metric will perform well large boards where 
    the perimieter(board) << area (board)
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    own_moves = game.get_legal_moves(player)
    own_movesCt = len(own_moves)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_movesCt = len(opp_moves)
    boardWidth=game.width
    boardHeight=game.height
    own_moves_dist = [calculateDistanceFromCenter(move,boardWidth,boardHeight,True) for move in own_moves]
    opp_moves_dist = [calculateDistanceFromCenter(move,boardWidth,boardHeight) for move in opp_moves]
    open_spaces =  len(game.get_blank_spaces())
    evalMetric = float(sum(own_moves_dist)*own_movesCt/((sum(opp_moves_dist)*opp_movesCt)+EPSILON)**2)/(open_spaces)
    return evalMetric

def custom_score_edgeAndCornerLimiting(game,player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    The score is calculated as the difference between legal moves available to
    the two players, normalized by the total number of blank spaces on the board.
    The intuition is that this metric gives higher (or lower) values later in the
    game than at the beginning for a given difference in legal moves as the 
    number of blank spaces decreases as the game progresses. This metric will perform well large boards where 
    the perimieter(board) << area (board)
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    own_moves = game.get_legal_moves(player)
    own_movesCt = len(own_moves)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_movesCt = len(opp_moves)
    edgeMoves = getEdges(game.width,game.height)
    cornerMoves = getCorners(game.width,game.height)
    edgeAndCornerMoves = edgeMoves+cornerMoves
    own_edgeAndCorner_moves = [move for move in own_moves if move in edgeAndCornerMoves]
    opp_edgeAndCorner_moves = [move for move in opp_moves if move in edgeAndCornerMoves]
    open_spaces =  len(game.get_blank_spaces())
    evalMetric = float(own_movesCt-len(own_edgeAndCorner_moves) - opp_movesCt+2*len(opp_edgeAndCorner_moves))/open_spaces
    #evalMetric = float(own_movesCt - opp_movesCt+3*len(opp_edgeAndCorner_moves))/open_spaces
    return evalMetric

def custom_score_normalizedByBlankSpaces(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    The score is calculated as the difference between legal moves available to
    the two players, normalized by the total number of blank spaces on the board.
    The intuition is that this metric gives higher (or lower) values later in the
    game than at the beginning for a given difference in legal moves as the 
    number of blank spaces decreases as the game progresses
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    open_spaces =  len(game.get_blank_spaces())
    evalMetric = float(own_moves - opp_moves)/open_spaces
    return evalMetric


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).
    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.
        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
       
        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        #legal_moves = game.get_legal_moves()
        #if there are more than 8 available legal moves, this is the player's
        #first move. So return the middle of the array (also board if the board
        #is empty - this needs to change with an openening book)
        if legal_moves == None:
            return None
        best_move = (-1,-1)
        #print(len(legal_moves))
        #print(legal_moves[len(legal_moves)/2])
        #if len(legal_moves)>8:
        #    best_move = legal_moves[len(legal_moves)/2]
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method=='minimax':
                if self.iterative:
                    d = 1
                    while True:
                        score,best_move = self.minimax(game,d)
                        d+=1
                else:
                    score,best_move = self.minimax(game,self.search_depth)
            elif self.method =='alphabeta':
                if self.iterative:
                    d = 1
                    while True:
                        score,best_move = self.alphabeta(game,d)
                        d+=1
                else:
                    score,best_move = self.alphabeta(game,self.search_depth)
            else:
                pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def is_terminal(self,game):
        """
        Checks if the game is in the terminal state
        Parameters:
        -----------
        game : isolation.Board
            An instance of he Isolation game 'Board' class representing the
            current game state
        Returns
        -------
        bool
            If no legal moves left, return True else False
        """
        return len(game.get_legal_moves())==0   
    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
                      
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if self.is_terminal(game) or depth ==0 or self.time_left()<2:
            return self.score(game,self),game.get_player_location(game.active_player)
        valid_moves = game.get_legal_moves()
        best_move = valid_moves[0]
        if maximizing_player:
            best_score = float("-Inf")
            for valid_move in valid_moves:
                score,mv = self.minimax(game.forecast_move(valid_move),depth-1,False)
                if score > best_score:
                    best_score = score
                    best_move = valid_move
        else:
            best_score = float("Inf")
            for valid_move in valid_moves:
                score,mv = self.minimax(game.forecast_move(valid_move),depth-1,True)
                if(score < best_score):
                    best_score = score
                    best_move = valid_move
        return best_score,best_move
  

        # TODO: finish this function!
        #raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # TODO: finish this function!
        if self.is_terminal(game) or depth ==0 or self.time_left()<2:
            return self.score(game,self),game.get_player_location(game.active_player)
        valid_moves = game.get_legal_moves()
        best_move = valid_moves[0]
        if maximizing_player:
            best_score = float("-Inf")
            for valid_move in valid_moves:
                score,mv = self.alphabeta(game.forecast_move(valid_move),depth-1,alpha,beta,False)
                alpha = max(alpha,score)
                if score > best_score:
                    best_score = score
                    best_move = valid_move
                if alpha >=beta:
                    break                
        else:
            best_score = float("Inf")
            for valid_move in valid_moves:
                score,mv = self.alphabeta(game.forecast_move(valid_move),depth-1,alpha,beta,True)
                beta = min(beta,score)
                if(score < best_score):
                    best_score = score
                    best_move = valid_move
                if alpha >=beta:
                    break
        return best_score,best_move