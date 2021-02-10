"""
ca.py

Introduction to Computational Science
Assignment 2 Developing a Unidimensional Cellular Automata
Daniël Mizrahi
10675418

This program generates all the states for a given unidimensional celullar
automata given the parameters r (the range), k (the alphabet, width and height
and rule number. The height determines the amount of steps that will be
performed. Also the intial state can be specified. Either it has value 1 in
the middle of the initial state or a random generated one. All these parameters
can be specified through the GUI.
"""
import numpy as np

from pyics import Model


def decimal_to_base_k(n, k):
    """
    Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1].

    Implemention based on algorithm provided in the assignment description.
    """
    base_k = list()
    while n != 0:
        remainder = n % k
        n = int(n / k)
        base_k.insert(0, remainder)
    return base_k


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)

        self.make_param('random_state_start', False)

    def setter_rule(self, val):
        """
        Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number.
        """
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """
        Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2.

        Uses decimal_to_base_k to build to rule set and prepends an array with
        0 to acquire the right length: k^(2r + 1).
        """
        base_k = decimal_to_base_k(self.rule, self.k)
        rule_size = self.k ** (2 * self.r + 1)
        self.rule_set = np.concatenate((np.zeros(rule_size - len(base_k),
                                        dtype=int), base_k))

    def check_rule(self, inp):
        """
        Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on.

        Determines the index of the input state needed to calculate the next
        value.
        """
        rule_index = 0
        pos = inp.size - 1
        for element in inp:
            rule_index = rule_index + int(element) * self.k ** pos
            pos = pos - 1
        return self.rule_set[(-rule_index - 1) % self.rule_set.size]

    def setup_initial_row(self):
        """
        Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k.

        In case a random array needs to generated a random array is generated
        otherwise the value in the middle of the array is set to 1 and the
        rest to 0.

        The random array toggle is set in random_state_start in __init__.
        """
        if self.random_state_start:
            return np.random.randint(self.k, size=self.width)
        else:
            split = int(self.width / 2)
            if self.width % 2 == 0:
                return np.array([0] * (split - 1) + [1] + [0] * split)
            else:
                return np.array([0] * split + [1] + [0] * split)

    def reset(self):
        """
        Initializes the configuration of the cells and converts the entered
        rule number to a rule set.
        """

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """
        Draws the current state of the grid.
        """

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                   cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """
        Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells.
        """
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                       for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
