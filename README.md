# FactorialHMM

FactorialHMM is a Python package for fast exact inference in Factorial Hidden Markov Models. 

FactorialHMM is freely available for academic use. A specific license must be obtained for any commercial or for-profit organization or for any web-diffusion purpose.

Our package allows:
* Simulating directly from the model 
* Simulating from the posterior distribution of states given the observations
* Calculating the (Viterbi) sequence of states with the largest posterior probability
* Calculating the Forward-Backward algorithm, and in particular likelihood of the data and the posterior probability (given all observations) of the marginal and joint state probabilities
as well as additional HMM-related procedures.

The running time and space requirement of all procedures is linear in the number of possible states. This package is highly modular, providing the user with maximal flexibility for developing downstream applications.

## Installation

Required Python 3+.

Simply download the `factorial_hmm.py` file, add its location to `sys.path` (e.g., `sys.path.append(path_to_dir)`), and import the library.

Prerequisites are `numpy` and `scipy`.

Comments are welcome at regevs@gmail.com or regev.schweiger@myheritage.com.

## Usage

The full documentation is available at the [Wiki](https://github.com/regevs/factorial_hmm/wiki) section.
