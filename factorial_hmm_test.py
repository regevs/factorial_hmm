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
    observed_indices = J = ['x1','x2']    
    
    def __init__(self, params, n_steps, calculate_on_init=True):
        # Call base class constructor
        super().__init__(params, n_steps, calculate_on_init)
        # In this example, m shifts the means for all hidden states
        m = self.params['m']
        # Draw means: one mean for each combination of hidden state and observation. Assume variance=1
        self.mus = m + np.random.RandomState().rand(*(self.hidden_indices.field_sizes + [len(self.observed_indices)]))
    
    def SetTransitionMatricesTensor(self):
        I = self.I
    
        for n in range(self.n_steps - 1):            
            self.SetTransitionSubmatrix(n, I.bit_a,[[0.2,0.6],[0.8,0.4]])
            self.SetTransitionSubmatrix(n, I.bit_b,[[0.3,0.5],[0.7,0.5]])
    
    def SetInitialHiddenState(self):
        I = self.I
        self.SetInitialHiddenSubstate(I.bit_a, [0.8, 0.2])
        self.SetInitialHiddenSubstate(I.bit_b, [0.4, 0.6])
    
    def GetObservedGivenHidden(self, observed_state, n_step):
        # Assume, for this example, that the two parts of the observation are independent
        a = np.ones(self.hidden_indices.field_sizes)
        for st in self.all_hidden_states:
            a[tuple(st)] = np.prod(scipy.stats.norm(loc=self.mus[list(st)+[Ellipsis]]).pdf(observed_state))
        return a
        
    def DrawObservedGivenHidden(self, hidden_state, n_step, random_state):
        return scipy.stats.norm(loc=self.mus[list(hidden_state)+[Ellipsis]]).rvs(size=len(self.observed_indices), random_state=random_state)
        



