import numpy as np

import itertools
import functools
import operator
import copy

class Indices(object):
    def __init__(self, fields_and_sizes):
        self.field_sizes = []
        self.fields = []
        for i, (field, size) in enumerate(fields_and_sizes):
            self.__dict__[field] = i
            self.fields.append(field)
            self.field_sizes.append(size)

    def __len__(self):
        return len(self.fields)

class FactorialHMM(object):
    def __init__(self, params, n_steps, calculate_on_init=True):
        self.params = params
        self.n_steps = n_steps

        self.observed_type = float   # unless otherwise specified downstream

        # TOOD: assert existence of self.hidden_indices
        self.n_hidden_states = len(self.hidden_indices)
        self.n_hidden_state_space = functools.reduce(operator.mul, self.hidden_indices.field_sizes)
        self.all_hidden_states = np.array(list(itertools.product(*[range(size) for size in self.hidden_indices.field_sizes])))
                
        self.transition_matrices_tensor = [[np.ones((size, size)) for size in self.hidden_indices.field_sizes] for n in range(n_steps-1)]

        # Preparation for multiply
        self.idx_in = np.arange(self.n_hidden_states, dtype=int).tolist()
        self.idx_out = np.zeros((self.n_hidden_states, self.n_hidden_states), dtype=int)
        for i in range(self.n_hidden_states):
            self.idx_out[i, :] = np.arange(self.n_hidden_states, dtype=int)
            self.idx_out[i, i] = int(self.n_hidden_states)
        self.idx_out = self.idx_out.tolist()

        self.initial_hidden_state_tensor = [np.ones(size) for size in self.hidden_indices.field_sizes]

        if calculate_on_init:
            self.SetTransitionMatricesTensor()
            self.SetInitialHiddenState()

    # Matrix_n[a,b] = P(z_{n+1}=a | z_{n}=b)
    def SetTransitionSubmatrix(self, n_step, field, submatrix):
        submatrix = np.array(submatrix, dtype=float)
        assert np.allclose(submatrix.sum(axis=0), 1)
        self.transition_matrices_tensor[n_step][field] = submatrix

    def SetInitialHiddenSubstate(self, field, substate):
        substate = np.array(substate, dtype=float)
        assert np.isclose(substate.sum(), 1)
        self.initial_hidden_state_tensor[field] = substate

    def GetTransitionMatrixColumn(self, n_step, prev_state):
        MS = self.transition_matrices_tensor[n_step]
        vecs = [M[:, prev_state[i]] for i, M in enumerate(MS)]
        return functools.reduce(np.kron, vecs)
        
    def GetTransitionMatrix(self, n_step):
        return functools.reduce(np.kron, self.transition_matrices_tensor[n_step])

    def MultiplyTransitionMatrixVector(self, n_step, vector, transpose=False):
        for i in range(self.n_hidden_states):
            submatrix = self.transition_matrices_tensor[n_step][i]

            vector = np.einsum(
                (submatrix.T if transpose else submatrix),
                [int(self.n_hidden_states), i],
                vector, 
                self.idx_in, 
                self.idx_out[i],
                )

        return vector

    def InitialHiddenStateConditional(self, initial_observed_state):
        initial_hidden_state = functools.reduce(np.kron, self.initial_hidden_state_tensor).reshape(self.hidden_indices.field_sizes)

        #for hidden_state in self.all_hidden_states:
            # * P(x_0|z_0)
            #initial_hidden_state[tuple(hidden_state)] *= self.obs_given_hidden[tuple(initial_observed_state) + tuple(hidden_state)]
            #initial_hidden_state[tuple(hidden_state)] *= self.GetObservedGivenHidden(initial_observed_state)[hidden_state]
        initial_hidden_state *= self.GetObservedGivenHidden(initial_observed_state, 0)

        return initial_hidden_state

    def SimulateHiddenStates(self, random_seed=0):
        rng = np.random.RandomState(random_seed)
        draw = lambda ps, rng=rng: rng.choice(len(ps), size=None, p=ps)

        # Allocate place for the sequence of hidden states
        Z = np.zeros((self.n_hidden_states, self.n_steps), dtype=int)

        # Draw the first hidden state set
        initial_hidden_state = functools.reduce(np.kron, self.initial_hidden_state_tensor)
        Z[:, 0] = self.all_hidden_states[draw(initial_hidden_state)]
        
        # Continue the hidden states
        for n_step in range(1, self.n_steps):
            i = draw(self.GetTransitionMatrixColumn(n_step-1, Z[:, n_step-1]))
            Z[:, n_step] = self.all_hidden_states[i]

        return Z

    def Simulate(self, random_seed=0):
        rng = np.random.RandomState(random_seed)
        #draw = lambda ps, rng=rng: rng.choice(len(ps), size=None, p=ps)

        Z = self.SimulateHiddenStates(random_seed)

        # Generate the observed states
        X = np.zeros((self.n_observed_states, self.n_steps), dtype=self.observed_type)
        for n_step in range(self.n_steps):
            # probs = self.obs_given_hidden[[Ellipsis] + list(Z[:, n_step])].ravel()
            # X[:, n_step] = self.all_observed_states[draw(probs)]
            X[:, n_step] = self.DrawObservedGivenHidden(Z[:, n_step], n_step, rng)

        return Z, X

    # Calculations from Bishop 13.2.4

    # This also calculates the likelihood of the observed
    def Forward(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[1]
        assert n_steps <= self.n_steps

        # Initialize
        alphas = np.ones(self.hidden_indices.field_sizes + [n_steps])
        scaling_constants = np.zeros(n_steps)

         # Alphas
        alphas[..., 0] = self.InitialHiddenStateConditional(observed_states[..., 0])
        scaling_constants[0] = alphas[..., 0].sum()
        alphas[..., 0] /= scaling_constants[0]

        for n_step in range(1, n_steps):
            #alphas[..., n_step] = self.obs_given_hidden[list(observed_states[..., n_step]) + [Ellipsis]]
            alphas[..., n_step] = self.GetObservedGivenHidden(observed_states[..., n_step], n_step)
            alphas[..., n_step] *= self.MultiplyTransitionMatrixVector(n_step-1, alphas[..., n_step-1])

            scaling_constants[n_step] = alphas[..., n_step].sum()
            alphas[..., n_step] /= scaling_constants[n_step]

        log_likelihood = np.log(scaling_constants).sum()

        return alphas, scaling_constants, log_likelihood

    def EStep(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[1]
        assert n_steps <= self.n_steps        

        # Forward
        alphas, scaling_constants, log_likelihood = self.Forward(observed_states)

        # Backward
        betas = np.ones(self.hidden_indices.field_sizes + [n_steps])
        
        for n_step in range(n_steps - 2, -1, -1):
            #vec = self.obs_given_hidden[list(observed_states[:, n_step+1]) + [Ellipsis]]
            vec = self.GetObservedGivenHidden(observed_states[..., n_step + 1], n_step + 1)
            betas[..., n_step] = self.MultiplyTransitionMatrixVector(n_step, vec * betas[..., n_step+1], transpose=True)
            betas[..., n_step] /= scaling_constants[n_step+1]

        # Join
        gammas = alphas * betas

        return alphas, betas, gammas, scaling_constants, log_likelihood

    # Notice Errata of page 628 of Bishop 13.2.4!!
    def CalculateXis(self, observed_states, alphas, betas, scaling_constants):
        # Slow, use CalculateAndCollapseXis if possible!!!

        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[1]

        xis = np.ones(self.hidden_indices.field_sizes + self.hidden_indices.field_sizes + [n_steps-1])
        for n_step in range(1, n_steps):
            xis[..., n_step-1] = (alphas[..., n_step-1].ravel()[:, np.newaxis] \
                #* self.obs_given_hidden[list(observed_states[:, n_step]) + [Ellipsis]].ravel()[np.newaxis, :] \
                * self.GetObservedGivenHidden(observed_states[..., n_step], n_step).ravel()[np.newaxis, :] \
                * self.GetTransitionMatrix(n_step - 1).T \
                * betas[..., n_step].ravel()[np.newaxis, :] \
                / scaling_constants[n_step]).reshape(self.hidden_indices.field_sizes + self.hidden_indices.field_sizes)

        return xis

    def CollapseGammas(self, gammas, fields):
        if not isinstance(fields, list):
            fields = [fields]
        else:
            fields = list(fields)

        return gammas.sum(tuple(set(range(self.n_hidden_states + 1)) - set(fields + [self.n_hidden_states])))

    def CollapseXis(self, xis, fields):
        # Slow, use CalculateAndCollapseXis if possible!!!
        if not isinstance(fields, list):
            fields = [fields]
        else:
            fields = list(fields)

        # Shortcuts
        return xis.sum(tuple(set(range(2 * self.n_hidden_states + 1)) - \
                            set(fields + [x + self.n_hidden_states for x in fields] + [2 * self.n_hidden_states])))


    def CalculateAndCollapseXis(self, field, observed_states, alphas, betas, scaling_constants):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[1]

        field_size = self.hidden_indices.field_sizes[field]

        # Create the matrix of collapsing to field 
        project = np.equal.outer(
            self.all_hidden_states[:, field], 
            np.arange(field_size)).reshape(self.hidden_indices.field_sizes + [field_size])

        collapsed_xis = np.ones((field_size, field_size, n_steps-1))
        idx_in1 = self.idx_in + [self.n_hidden_states]
        idx_in2 = self.idx_in + [self.n_hidden_states + 1]
        idx_out = [self.n_hidden_states, self.n_hidden_states+1]

        for n_step in range(1, n_steps):
            #b = self.obs_given_hidden[list(observed_states[:, n_step]) + [Ellipsis]] * betas[..., n_step]
            b = self.GetObservedGivenHidden(observed_states[..., n_step], n_step) * betas[..., n_step]

            bp = b[..., np.newaxis] * project

            a = alphas[..., n_step-1]
            ap = a[..., np.newaxis] * project

            mbp = np.stack([self.MultiplyTransitionMatrixVector(n_step - 1, bp[..., i], transpose=True) for i in range(field_size)], axis=-1)

            collapsed_xis[:, :, n_step-1] = np.einsum(
                ap,
                idx_in1,
                mbp,
                idx_in2,
                idx_out
                )

            collapsed_xis[:, :, n_step-1] /= scaling_constants[n_step]

        return collapsed_xis

    def GetXiRow(self, hidden_state_row, n_step, observed_states, alphas, betas, scaling_constants):
        basis_vector = np.zeros(self.hidden_indices.field_sizes)
        basis_vector[tuple(hidden_state_row)] = 1

        aa = (basis_vector * alphas[..., n_step]) 

        maa = self.MultiplyTransitionMatrixVector(n_step, aa, transpose=False)

        bb = betas[..., n_step+1]  \
            * self.GetObservedGivenHidden(observed_states[..., n_step+1], n_step + 1) \
            / scaling_constants[n_step+1]
        
        xi = maa * bb

        return xi 

    def ViterbiSlower(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[1]

        back_pointers = np.ones((self.n_hidden_state_space, n_steps-1), dtype=int)   
        lls = np.ones((self.n_hidden_state_space, n_steps), dtype=float)   

        ll = np.log(self.InitialHiddenStateConditional(observed_states[:, 0])).reshape((self.n_hidden_state_space))
        lls[:, 0] = ll

        for n_step in range(1, n_steps):
            M = np.log(self.GetTransitionMatrix(n_step-1)) + ll[np.newaxis, :]
            back_pointers[:, n_step-1] = np.argmax(M, axis=1)

            #ll = np.log(self.obs_given_hidden[list(observed_states[:, n_step]) + [Ellipsis]]).reshape((self.n_hidden_state_space))
            ll = np.log(self.GetObservedGivenHidden(observed_states[..., n_step], n_step)).reshape((self.n_hidden_state_space))
            ll += M[np.arange(self.n_hidden_state_space), back_pointers[:, n_step-1]]
            lls[:, n_step] = ll

        # Backtrack
        most_likely = np.zeros(n_steps, dtype=int)
        most_likely[n_steps-1] = np.argmax(ll)
        last = most_likely[n_steps-1]

        for n_step in range(n_steps-1, 0, -1):
            most_likely[n_step-1] = back_pointers[last, n_step-1]
            last = most_likely[n_step-1]

        return most_likely, back_pointers, lls

    def Viterbi(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[1]

        mgrid_prepared = list(np.mgrid[[range(s) for s in self.hidden_indices.field_sizes]])
        
        #back_pointers = np.ones((self.n_hidden_state_space, n_steps-1), dtype=int)   
        back_pointers = np.ones(self.hidden_indices.field_sizes + [self.n_hidden_states] + [n_steps-1], dtype=int)   
        lls = np.zeros(self.hidden_indices.field_sizes + [n_steps], dtype=float)

        lls[..., 0] = np.log(self.InitialHiddenStateConditional(observed_states[:, 0]))
        ll = lls[..., 0]

        for n_step in range(1, n_steps):
            vector = ll.copy()
            argmaxes = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states], dtype=int)
            #pointers = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states], dtype=int)

            #print(vector)
            #print(np.log(self.transition_matrices_tensor[n_step-1][0]))
            #print(np.log(self.transition_matrices_tensor[n_step-1][1]))

            for i in range(self.n_hidden_states):
                submatrix = np.nan_to_num(np.log(self.transition_matrices_tensor[n_step-1][i][:, :]))

                idx_a = [slice(None)] + [np.newaxis] * self.n_hidden_states
                idx_a[i+1] = slice(None)

                B = submatrix[idx_a] + vector[np.newaxis, ...]

                # TOOD: unify max and argmax
                vector = np.moveaxis(B.max(axis=i+1), 0, i)
                mx = np.moveaxis(B.argmax(axis=i+1), 0, i)

                argmaxes[..., i] = mx
                # grids = np.meshgrid(*[np.arange(s) for s in self.hidden_indices.field_sizes])
                # print([x.shape for x in grids])
                # print(mx.shape)
                # #grids[i] = mx
                # #argmaxes[..., :i] = argmaxes[]
                # print(argmaxes[grids].shape)
                
                # print(vector)
                # print(mx)
                # #print(argmaxes[..., :i])

            #print(argmaxes.reshape(-1, self.n_hidden_states))

            back_pointers[..., self.n_hidden_states - 1, n_step - 1] = argmaxes[..., self.n_hidden_states - 1]
            for i in range(self.n_hidden_states - 2, -1, -1):
                indices = mgrid_prepared.copy()
                indices[i+1] = argmaxes[..., i+1]
                back_pointers[..., i, n_step - 1] = argmaxes[..., i][indices]
            
            #M = np.log(self.GetTransitionMatrix(n_step-1)) + ll[np.newaxis, :].reshape((self.n_hidden_state_space))
            

            # print(vector.ravel())
            # print(np.max(M, axis=1))
            # print(np.allclose(vector.ravel(), np.max(M, axis=1)))
            # return

            #back_pointers[:, n_step-1] = np.argmax(M, axis=1)
            #print(back_pointers[:, n_step-1])
            #A = np.array(np.unravel_index(np.argmax(M, axis=1), self.hidden_indices.field_sizes)).T.reshape(self.hidden_indices.field_sizes + [self.n_hidden_states])
            #print(np.all(A == pointers))
            #return 

            #ll = np.log(self.obs_given_hidden[list(observed_states[:, n_step]) + [Ellipsis]])
            ll = np.log(self.GetObservedGivenHidden(observed_states[..., n_step], n_step))
            ll += vector #M[np.arange(2**len(I)), back_pointers[:, n_step-1]]
            lls[..., n_step] = ll

            #return

        # Backtrack
        most_likely = np.zeros((self.n_hidden_states, n_steps), dtype=int)
        most_likely[..., n_steps - 1] = np.unravel_index(np.argmax(ll.ravel()), self.hidden_indices.field_sizes)
        last = most_likely[..., n_steps - 1]

        for n_step in range(n_steps - 1, 0, -1):
            # print(n_step)
            # print(most_likely[..., n_step - 1])
            # print(back_pointers.shape, [last.tolist() + [slice(None), n_step-1]])
            most_likely[..., n_step - 1] = back_pointers[last.tolist() + [slice(None), n_step-1]]
            last = most_likely[..., n_step - 1]

        return most_likely, back_pointers, lls    

    def CalculateJointLogLikelihood(self, observed_states, hidden_states):
        n_steps = observed_states.shape[1]

        logp = 0.0
        # p(x0,z0)
        logp += np.log(self.InitialHiddenStateConditional(observed_states[..., 0])[tuple(hidden_states[..., 0])])

        for n_step in range(1, n_steps):
            # p(z_n | z_{n-1})
            col = self.GetTransitionMatrixColumn(n_step-1, hidden_states[..., n_step-1])
            logp += np.log(col[np.ravel_multi_index(hidden_states[..., n_step], self.hidden_indices.field_sizes)])

            # p(x_n | z_n)
            logp += np.log(self.GetObservedGivenHidden(observed_states[..., n_step], n_step)[tuple(hidden_states[..., n_step])])

        return logp

    def DrawFromPosterior(self, observed_states, alphas, betas, scaling_constants, start=0, end=None, random_seed=None):
        if end == None:
            end = self.n_steps

        rng = np.random.RandomState(random_seed)
        draw = lambda ps, rng=rng: rng.choice(len(ps), size=None, p=ps)

        # Allocate place for the sequence of hidden states
        Z = np.zeros((self.n_hidden_states, end-start), dtype=int)

        # Draw the first hidden state set
        initial_hidden_state_posterior = alphas[..., start] * betas[..., start]
        initial_hidden_state_posterior /= initial_hidden_state_posterior.sum()

        Z[:, 0] = self.all_hidden_states[draw(initial_hidden_state_posterior.ravel())]

        # Continue the hidden states
        for n_step in range(start+1, end):
            xi_row = self.GetXiRow(Z[:, n_step-1-start], n_step-1, observed_states, alphas, betas, scaling_constants)
            xi_row /= xi_row.sum()

            Z[:, n_step-start] = self.all_hidden_states[draw(xi_row.ravel())]

        return Z

    def DescaleAlphasBetas(self, alphas, betas, scaling_constants):
        acc_scaling_constants = np.multiply.accumulate(scaling_constants)

        acc_scaling_constants_from_back = np.multiply.accumulate(scaling_constants[::-1])[::-1]
        acc_scaling_constants_from_back[:-1] = acc_scaling_constants_from_back[1:]
        acc_scaling_constants_from_back[-1] = 1.0
        
        descaled_alphas = alphas * acc_scaling_constants[(np.newaxis,) * (len(betas.shape) - 1) + (slice(None),)]
        descaled_betas = betas * acc_scaling_constants_from_back[(np.newaxis,) * (len(betas.shape) - 1) + (slice(None),)]

        return descaled_alphas, descaled_betas

    def SetTransitionMatricesTensor(self):
        raise NotImplementedError

    def SetInitialHiddenState(self):
        raise NotImplementedError

    def GetObservedGivenHidden(self, observed_state):
        raise NotImplementedError

    def DrawObservedGivenHidden(self, hidden_state, n_step, random_state):
        raise NotImplementedError


class FactorialHMMDiscreteObserved(FactorialHMM):
    def __init__(self, params, n_steps, calculate_on_init=True):
        # Call base class constructor
        super().__init__(params, n_steps, calculate_on_init)

        self.observed_type = int

        self.n_observed_states = len(self.observed_indices)
        self.n_observed_state_space = functools.reduce(operator.mul, self.observed_indices.field_sizes)
        self.all_observed_states = np.array(list(itertools.product(*[range(size) for size in self.observed_indices.field_sizes])))

        self.obs_given_hidden = np.ones(self.observed_indices.field_sizes + self.hidden_indices.field_sizes + [self.n_steps])

        if calculate_on_init:
            self.SetObservedGivenHidden()

    def SetObservedGivenHiddenSubmatrix(self, n_step, obs_given_hidden):
        self.obs_given_hidden[..., n_step] = obs_given_hidden            

    def SetObservedGivenHidden(self):
        # return # Prepare the matrix of P(X=x|Z=z)
        raise NotImplementedError

    def GetObservedGivenHidden(self, observed_state, n_step):
        # Returns a tensor of P(x|z), per z
        return self.obs_given_hidden[list(observed_state) + [Ellipsis, n_step]]

    def DrawObservedGivenHidden(self, hidden_state, n_step, random_state):
        draw = lambda ps: random_state.choice(len(ps), size=None, p=ps)
        probs = self.obs_given_hidden[[Ellipsis] + list(hidden_state) + [n_step]].ravel()
        return self.all_observed_states[draw(probs)]


class FactorialHMMGeneralObserved(FactorialHMM):
    def __init__(self, params, n_steps, calculate_on_init=True):
        # Call base class constructor
        super().__init__(params, n_steps, calculate_on_init)

        self.n_observed_states = len(self.observed_indices)
        

class FullDiscreteFactorialHMM(FactorialHMMDiscreteObserved):
    def __init__(self, params, n_steps, calculate_on_init=False):
        # First initialize the hidden and observed indices
        assert 'hidden_alphabet_size' in params.keys(), "params dictionary must contain 'hidden_alphabet_size':<alphabet size>"
        assert 'n_hidden_states' in params.keys(), "params dictionary must contain 'n_hidden_states':<number of hidden chains>"
        assert 'observed_alphabet_size' in params.keys(), "params dictionary must contain 'observed_alphabet_size':<alphabet size>"
        assert 'n_observed_states' in params.keys(), "params dictionary must contain 'n_observed_states':<number of observed chains>"

        self.hidden_indices = self.I = Indices([['z{}'.format(i), params['hidden_alphabet_size']] for i in range(params['n_hidden_states'])])
        self.observed_indices = self.J = Indices([['x{}'.format(i), params['observed_alphabet_size']] for i in range(params['n_observed_states'])])

        # Call base class constructor
        super().__init__(params, n_steps, calculate_on_init)

    def SetTransitionMatricesTensor(self):
        for n in range(self.n_steps - 1):
            for field in range(self.n_hidden_states):
                self.SetTransitionSubmatrix(n, field, self.params['transition_matrices'][field, :, :])

    def SetInitialHiddenState(self):
        for field in range(self.n_hidden_states):
            self.SetInitialHiddenSubstate(field, self.params['initial_hidden_state'][field, :])

    def SetObservedGivenHidden(self):        
        for n_step in range(self.n_steps):
            self.SetObservedGivenHiddenSubmatrix(n_step, self.params['obs_given_hidden'])            

    def MStep(self, observed_states, alphas, betas, gammas, scaling_constants):
        # Shortcuts
        I = self.hidden_indices
        J = self.observed_indices
        P = self.params

        K = P['hidden_alphabet_size']

        initial_hidden_state_estimate = np.zeros((self.n_hidden_states, K))   
        transition_matrices_estimates = np.zeros((self.n_hidden_states, K, K))        
        for field in range(self.n_hidden_states):
            collapsed_xis = self.CalculateAndCollapseXis(field, observed_states, alphas, betas, scaling_constants)
            collapsed_gammas = self.CollapseGammas(gammas, field)

            initial_hidden_state_estimate[field, :] = collapsed_gammas[:, 0] / collapsed_gammas[:, 0].sum()
            transition_matrices_estimates[field, :, :] = (collapsed_xis.sum(axis=2) / collapsed_gammas[:, :-1].sum(axis=1)[:, np.newaxis]).T

        emission_probability_estimate = np.zeros(self.observed_indices.field_sizes + self.hidden_indices.field_sizes)
        for hid_state in self.all_hidden_states:                
            gammas_at_hid = gammas[tuple(hid_state) + (Ellipsis,)]

            for obs_state in self.all_observed_states:
                indices = np.where(np.all(observed_states == obs_state[:, np.newaxis], axis=0))[0]            
                emission_probability_estimate[tuple(obs_state) + tuple(hid_state)] = gammas_at_hid[indices].sum() / gammas_at_hid.sum()
                
       
        new_params = copy.deepcopy(P)
        new_params['initial_hidden_state'] = initial_hidden_state_estimate
        new_params['transition_matrices'] = transition_matrices_estimates
        new_params['obs_given_hidden'] = emission_probability_estimate

        return new_params

    def EM(self, observed_states, n_iterations=1e8, likelihood_precision=1e-10, verbose=False, print_every=1, random_seed=None):
        old_log_likelihood = -np.inf
        n_iter = 0
        random_state = np.random.RandomState(random_seed)

        # Create an HMM object with an initial random state
        params = copy.deepcopy(self.params)
        K = params['hidden_alphabet_size']

        for field in range(params['n_hidden_states']):
            M = random_state.random_sample((K,K))
            M /= M.sum(axis=0)[np.newaxis, :]
            params['transition_matrices'][field, :, :] = M

            S = random_state.random_sample(K)
            S /= S.sum()
            params['initial_hidden_state'][field, :] = S

        G = random_state.random_sample(self.observed_indices.field_sizes + self.hidden_indices.field_sizes)
        G /= G.sum(axis=tuple(np.arange(self.n_observed_states)))[(np.newaxis,) * self.n_observed_states + (Ellipsis,)]
        params['obs_given_hidden'] = G

        H = FullDiscreteFactorialHMM(params, self.n_steps, True)

        while True:
            alphas, betas, gammas, scaling_constants, log_likelihood = H.EStep(observed_states)
            new_params = H.MStep(observed_states, alphas, betas, gammas, scaling_constants)

            if verbose and (n_iter%print_every == 0 or n_iter == n_iterations-1):
                print("Iter: {}\t LL: {}".format(n_iter, log_likelihood))

            n_iter += 1
            if n_iter == n_iterations:
                break

            if np.abs(log_likelihood - old_log_likelihood) < likelihood_precision:
                break

            old_log_likelihood = log_likelihood
            H = FullDiscreteFactorialHMM(new_params, self.n_steps, True)

        return H




