class DOS:

    def __init__(self, spin, h=0., mu=0.):

        self.mu = mu

        if spin == 'up':

            self.h = -h

        elif spin == 'down':

            self.h = h

    def constDOS(self):

        # want to have a minimum of if statements inside a function call

        return lambda eps: 1. if -1. + self.mu + self.h <= eps <= 1. + self.mu + self.h else 0.