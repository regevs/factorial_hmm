# import importlib
# import factorial_hmm
# importlib.reload(factorial_hmm)
from factorial_hmm import *

import scipy.stats

class BasicFactorialHMM(FactorialHMMDiscreteObserved):
    hidden_indices = I = Indices([['bit', 2], ['trit', 3], ['quit', 4]])
    observed_indices = J = Indices([['bit_a', 2], ['bit_b', 2], ['digit', 10]])

    def SetTransitionMatricesTensor(self):
        I = self.I
        p = self.params['p']
        p2 = self.params['p2']

        for n in range(self.n_steps - 1):
            self.SetTransitionSubmatrix(n, I.bit, [[1-p, p], [p, 1-p]])
            self.SetTransitionSubmatrix(n, I.trit, [[1-p, p*(1-p2), 1/3], [p*p2, 1-p ,1/3], [p*(1-p2), p*p2, 1/3]])
            self.SetTransitionSubmatrix(n, I.quit, np.identity(4))

    def SetInitialHiddenState(self):
        I = self.I

        self.SetInitialHiddenSubstate(I.bit, [0.8, 0.2])
        self.SetInitialHiddenSubstate(I.trit, [0.2, 0.4, 0.4])
        self.SetInitialHiddenSubstate(I.quit, [0.25, 0.25, 0.25, 0.25])

    def SetObservedGivenHidden(self):
        # Prepare the matrix of P(X=x|Z=z)
        obs_given_hidden = np.ones(self.observed_indices.field_sizes + self.hidden_indices.field_sizes)

        random_state = np.random.RandomState(0)
        for st in self.all_hidden_states:
            R = random_state.rand(*self.observed_indices.field_sizes)
            R /= R.sum()
            obs_given_hidden[[Ellipsis] + list(st)] = R

        for n_step in range(self.n_steps):
            self.SetObservedGivenHiddenSubmatrix(n_step, obs_given_hidden)


class GaussianFactorialHMM(FactorialHMMGeneralObserved):
    hidden_indices = I = Indices([['bit_a', 2], ['bit_b', 2]])
    observed_indices = J = ['height']

    def __init__(self, params, n_steps, calculate_on_init=True):
        # Call base class constructor
        super().__init__(params, n_steps, calculate_on_init)
        self.mus = 5*np.mgrid[[range(s) for s in self.hidden_indices.field_sizes]].sum(axis=0).astype(float)        
        self.rvs = scipy.stats.norm(loc=self.mus)

    def SetTransitionMatricesTensor(self):
        I = self.I
        p = self.params['p']

        for n in range(self.n_steps - 1):
            self.SetTransitionSubmatrix(n, I.bit_a, [[1-p, p], [p, 1-p]])
            self.SetTransitionSubmatrix(n, I.bit_b, [[1-p, p], [p, 1-p]])

    def SetInitialHiddenState(self):
        I = self.I

        self.SetInitialHiddenSubstate(I.bit_a, [0.8, 0.2])
        self.SetInitialHiddenSubstate(I.bit_b, [0.1, 0.9])

    def GetObservedGivenHidden(self, observed_state, n_step):
        return self.rvs.pdf(observed_state[0])

    def DrawObservedGivenHidden(self, hidden_state, n_step, random_state):
        return scipy.stats.norm(loc=self.mus[tuple(hidden_state)]).rvs(size=1, random_state=None)
        

