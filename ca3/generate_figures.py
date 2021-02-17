"""
ca.py

Introduction to Computational Science
Assignment 3 The λ Parameter and Sampling
Daniël Mizrahi
10675418

This program generates all the states for a given unidimensional celullar
automata given the parameters r (the range), k (the alphabet, width and height
and the langton parameter to generate a rule table. The height determines the
amount of steps that will be performed. Also the intial state can be
specified. Either it has value 1 in the middle of the initial state or a
random generated one. This version has the GUI removed as to generate only
results and to then plot these results.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def H(count):
    """
    Will calculate the Shannon entropy for a given row. Not only a singular
    cell will be taken in consideration but also it's neighbourhood as
    described in the slides.
    """
    count = count / np.sum(count)
    entropy_sum = 0
    for p in count:
        if p == 0.0:
            continue
        else:
            entropy_sum = entropy_sum + p * np.log2(p)
    return -entropy_sum


def calculate_langton(rule_set):
    """
    Will calculate the langton value for a given rule_set
    """
    n = 0
    for rule in rule_set:
        if rule == 0:
            n = n + 1
    return (rule_set.size - n) / rule_set.size


class CASimNoGUI():
    def __init__(self):
        self.t = 0
        self.rule_set = []
        self.config = None

        self.r = 1
        self.k = 2
        self.width = 50
        self.height = 50

        self.random_state_start = True
        self.langton = 0.0

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
        self.langton_to_table()

    def step(self):
        """
        Performs a single step of the simulation by advancing time (and thus
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

    def setter_langton(self, langton):
        """
        Makes sure the langton parameter value is smaller or equal to 1.0 or
        larger or equal to 0.0. Not al parameter values are relevant. It's
        only relevant if it's a viable discrete step size in relation to r and
        k. The function rounds the value accordingly.
        """
        if langton < 0.0:
            self.langton = 0.0
        if langton > 1.0:
            self.langton = 1.0
        rule_size = self.k ** (2 * self.r + 1)
        step_size = 1 / rule_size
        remainder = langton % step_size
        if remainder / step_size < 0.5:
            self.langton = langton - remainder
        else:
            self.langton = langton - remainder + step_size

    def langton_to_table(self):
        """
        Generates a rule table set according to a given langton parameter
        value. The algorithm used is the Table-Walk-Through algorithm as
        described in the paper COMPUTATION AT THE EDGE OF CHAOS written by
        Chris Langton.
        
        If the simulation just started it makes sure the starting rule_set is
        set to an array with only zeros with the right length. If perhaps new
        values are set for r and k the rule_set will be reinitialized to to an
        array with only zeros to start over again. In case only the
        langton parameter is adjusted it will be generated based on the
        previous rule table.
        """
        rule_size = self.k ** (2 * self.r + 1)
        if np.array_equal(self.rule_set, []):
            self.rule_set = np.zeros(rule_size, dtype=int)
        if rule_size != self.rule_set.size:
            self.rule_set = np.zeros(rule_size, dtype=int)
        langton_step_size = self.langton - calculate_langton(self.rule_set)
        select = int(np.round_((self.langton - calculate_langton(self.rule_set)) * rule_size))
        if select > 0:
            for i in np.random.choice(np.where(self.rule_set == 0)[0], select, replace=False):
                self.rule_set[i] = np.random.choice(np.arange(1, self.k))
        elif select < 0:
            for i in np.random.choice(np.where(self.rule_set != 0)[0], np.abs(select), replace=False):
                self.rule_set[i] = 0

    def calculate_entropy(self):
        entropy_sum = 0
        for t in range(self.height):
            rule_size = self.k ** (2 * self.r + 1)
            count = np.zeros(rule_size, dtype=int)
            for patch in range(self.width):
                indices = [i % self.width
                           for i in range(patch - self.r, patch + self.r + 1)]
                values = self.config[t, indices]
                rule_index = 0
                pos = values.size - 1
                for element in values:
                    rule_index = rule_index + int(element) * self.k ** pos
                    pos = pos - 1
                count[rule_index] = count[rule_index] + 1
            entropy_sum = entropy_sum + H(count)
        return entropy_sum / self.height
            

if __name__ == '__main__':
    sim = CASimNoGUI()
    sim.r = 2
    sim.k = 4
    sim.height = 10000
    sim.width = 100
    results = np.empty((10, 21))
    for simulation in range(10):
        print('simulation ' + str(simulation + 1))
        langton = 0.0
        sim.setter_langton(langton)
        for increment in range(21):
            print('langton: ' + str(langton))
            sim.reset()
            sim.step()
            entropy = sim.calculate_entropy()
            print('entropy: ' + str(entropy))
            langton = langton + 0.05
            sim.setter_langton(langton)
            results[simulation][increment] = entropy
    print('simulation finished...')
    
    l = np.arange(0.0, 1.05, step=0.05)
    fig, ax = plt.subplots()
    ax.set(xlabel='langton (λ)', ylabel='entropy (H)',
           title='Assignment 3: \nEntropy in relation to λ')
    ax.grid()
    ax.set_xticks(l)
    caption = 'This is a line plot for a total of 10 rule tables of which ' \
              'the the langton value is increased with 0.05 as steps. ' \
              'For every langton value the Shannon entropy is calculated by' \
              ' calculating the average entropy for every step t. r = 2 and' \
              ' k = 4. The height and therefore t is 10^4. The width = 100.' \
              ' All starting positions are random. The quality of the ' \
              'results could be improved by decreasing ' \
              ' the langton step value and performing more simulations'
    fig.text(.5, 0.025, caption, ha='center', wrap=True)
    for result in results:
        ax.plot(l, result)
    plt.show()
