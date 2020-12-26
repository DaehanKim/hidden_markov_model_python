# Hidden Markov Model(HMM)

This repo is a toy python implementation of [A Revealing Introduction to Hidden Markov Model](http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf). This implementation of HMM uses scaled alpha and beta to avoid numerical underflow resulting from repeatitive multiplication of probabilities. For a case of multiple sequences being observed, multiple sequence fitting is also implemented as shown in [baum-welch wikipedia](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)


### Notes

 - This repo was initially developed for a Knowledge Tracing task, so the number of observation types and hidden state types are set to 2. You can easily adjust these to arbitrary integers with minor fixes. 
 - `config.py` is a setting for initial parameters (transition matrix, emission matrix, initial state probability) though it is described in knowledge tracing terms.
 - Run with `python HMM.py`
