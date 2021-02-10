"""
ca_figures.py

Introduction to Computational Science
Assignment 2 Developing a Unidimensional Cellular Automata
Daniël Mizrahi
10675418

I have decided to use ca.py as a base for the script to generate my figures.
All graphical related methods are removed. The class attributes are regular
Python class attributes. step() has been modified to perform all steps instead
of one at a time. Apart from new methods added those are the only changes.
Since the simulation is very computational intensive and will take a lot of
time to complete a GUI is cumbersome.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


class CASimNoGUI():
    def __init__(self):

        self.t = 0
        self.rule_set = []
        self.config = None

        self.r = 1
        self.k = 2
        self.width = 50
        self.height = 10000
        self.rule = 1

        self.random_state_start = False

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

    def step(self):
        """
        Performs all steps of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells.
        """
        while True:
            self.t += 1
            if self.t >= self.height:
                return True
            else:
                for patch in range(self.width):
                    # We want the items r to the left and to the right of this
                    # patch, while wrapping around (e.g. index -1 is the last
                    # item on the row). Since slices do not support this, we
                    # create an array with the indices we want and use that to
                    # index our grid.
                    indices = [i % self.width
                            for i in range(patch - self.r, patch + self.r + 1)]
                    values = self.config[self.t - 1, indices]
                    self.config[self.t, patch] = self.check_rule(values)

    def values(self, patch, t):
        """
        Like step it return a set of values but not to calculate the next step
        but in order for brent to compare values.
        """
        indices = [i % self.width
                   for i in range(patch - self.r, patch + self.r + 1)]
        values = self.config[t, indices]
        return values

    def brent(self, patch):
        """
        Brent is an algorithm to find cycles. Implemention inspired based on
        the code provided here:
        https://en.wikipedia.org/wiki/Cycle_detection#Brent's_algorithm
        """
        power = lam = 1
        tortoise = 0
        hare = 1
        while np.array_equal(self.values(patch, tortoise),
                             self.values(patch, hare)) is False:
            if power == lam:
                tortoise = hare
                power = power * 2
                lam = 0
            hare = hare + 1
            lam = lam + 1
            if hare >= self.height or tortoise >= self.height:
                break

        tortoise = 0
        hare = lam

        mu = 0
        while np.array_equal(self.values(patch, tortoise),
                             self.values(patch, hare)) is False:
            tortoise = tortoise + 1
            hare = hare + 1
            mu = mu + 1
            if hare >= self.height or tortoise >= self.height:
                break

        return lam, mu

    def average_cycle_length(self):
        """
        Calculates the average cycle length of the whole automaton using brent.
        """
        sum_avg = 0
        for patch in range(self.width):
            sum_avg = sum_avg + self.brent(patch)[0]
        return sum_avg / self.width


if __name__ == '__main__':
    sim = CASimNoGUI()

    labels = [rule for rule in range(256)]
    x_axis = np.arange(256)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_ylabel('Average cycle length')
    ax.set_xlabel('Rule')
    ax.set_xticklabels(labels)
    title = 'Assignment 2 Developing a Unidimensional Cellular Automata '

    sim.random_state_start = False
    average_cycle_lengths = list()
    sim.width = 100
    for rule in range(256):
        print('random:' + str(sim.random_state_start) + ' rule:' + str(rule))
        sim.rule = rule
        sim.reset()
        sim.step()
        avg = sim.average_cycle_length()
        average_cycle_lengths.append(avg)
        print('average_cycle_length:' + str(avg))
    rect = ax.bar(x_axis, average_cycle_lengths)
    ax.set_title(title + 'Non random initial state')
    fig.tight_layout()
    caption = 'This is a bar chart plot of the average cycle length for a ' \
              'given rule. k: 2, r: 1, Height: 10^4, Width: 100. In this ' \
              'case the initial state is not random. So there is no need to' \
              ' execute the simulation multiple times since every ' \
              'simulation is the exact same. The initial state has only ' \
              'the value at index 49 set to 1 and the rest to 0. ' \
              'Unfortunately because of time constraints I have decided to ' \
              'use a width of 100 instead of something larger. This makes ' \
              'these tests less reliable.'
    fig.text(.5, .05, caption, ha='center', wrap=True)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_ylabel('Average cycle length')
    ax.set_xlabel('Rule')
    ax.set_xticklabels(labels)
    title = 'Assignment 2 Developing a Unidimensional Cellular Automata '

    sim.random_state_start = True
    average_cycle_lengths = list()
    sim.width = 50
    for rule in range(256):
        print('random:' + str(sim.random_state_start) + ' rule:' + str(rule))
        sim.rule = rule
        avg = 0
        for trail in range(3):
            print('trail: ' + str(trail))
            sim.reset()
            sim.step()
            avg = avg + sim.average_cycle_length()
        average_cycle_lengths.append(avg / 3)
        print('average_cycle_length:' + str(avg))
    rect = ax.bar(x_axis, average_cycle_lengths)
    ax.set_title(title + 'Random initial state')
    fig.tight_layout()
    caption = 'This is a bar chart plot of the average cycle length for a ' \
              'given rule. k: 2, r: 1, Height: 10^4, Width: 50 In this case' \
              ' the initial state is random and the simulation is performed' \
              ' 3 times for a given rule to achieve an average. ' \
              'Unfortunately because of time constraints I have decided to ' \
              'use a width of 50 instead of something larger. Same goes for' \
              ' the number of trails for a given rule. This makes these ' \
              'tests less reliable.'
    fig.text(.5, .05, caption, ha='center', wrap=True)
    plt.show()
