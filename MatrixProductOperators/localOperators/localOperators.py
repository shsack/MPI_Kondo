import numpy as np

class localOperators:

    """defines the local operators in terms of the occupation basis : {/0>, /up>, /down>, /up,down>}"""

    def __init__(self):

        self.I = np.identity(4)
        self.Z = np.zeros((4, 4))
        self.create_up = np.array([[0, 0, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 1, 0]])

        self.create_down = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0]])

        self.annih_up = np.array([[0, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 0]])

        self.annih_down = np.array([[0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]])

        self.n_up = np.array([[0, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 1]])

        self.n_down = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        self.n_up_down = np.array([[0, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 2]])
