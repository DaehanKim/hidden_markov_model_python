import numpy as np
from config import *

class HMM_SINGLE :
    '''
    HMM for single sequence only
    '''
    def __init__(self):
        self.prior_knowledge =  np.array([1-KNOW_INITIAL, KNOW_INITIAL])
        self.transition_mat = np.array([[1-LEARN_INITIAL, LEARN_INITIAL],
                            [FORGET_INITIAL, 1-FORGET_INITIAL]])
        self.emission_mat = np.array([[1-GUESS_INITIAL, GUESS_INITIAL],
                            [SLIP_INITIAL, 1-SLIP_INITIAL]])

        self.alpha = None
        self.beta = None
        self.c_arr = None
        self.gamma = None
        self.xi = None


    def forward(self, sequence):
        ''' compute alpha '''
        self.alpha = np.zeros((len(sequence), 2))
        self.c_arr = []

        c0 = 0
        for i in range(2):
            self.alpha[0,i] = self.prior_knowledge[i] * self.emission_mat[i, sequence[0]]
            c0 += self.alpha[0,i]
        c0 = 1/c0
        self.c_arr.append(c0)

        # scale alpha zero
        self.alpha[0] = c0 * self.alpha[0]

        for t in range(1,len(sequence)):
            ct = 0
            for i in range(2):
                self.alpha[t,i] = 0
                for j in range(2):
                    self.alpha[t,i] += self.alpha[t-1,j]*self.transition_mat[j,i]
                self.alpha[t,i] = self.alpha[t,i]*self.emission_mat[i,sequence[t]]
                ct += self.alpha[t,i]
            ct = 1/ct
            self.c_arr.append(ct)
            self.alpha[t] *= ct

    def backward(self, sequence):
        '''compute beta'''

        T = len(sequence)
        self.beta = np.zeros((T, 2))
        self.beta[T-1] = np.full((2,), self.c_arr[T-1], dtype=np.float)

        for t in range(T-2, -1,-1):
            for i in range(2):
                self.beta[t,i] = 0
                for j in range(2):
                    self.beta[t,i] += self.transition_mat[i,j]*self.emission_mat[j,sequence[t]] * self.beta[t+1, j]
                self.beta[t,i] *= self.c_arr[t]


    def get_gamma_xi(self, sequence):
        T = len(sequence)
        self.xi = np.zeros((T-1,2,2))
        self.gamma = np.zeros((T,2))
        
        for t in range(T-1):
            for i in range(2):
                self.gamma[t, i] = 0 
                for j in range(2):
                    self.xi[t,i,j] = self.alpha[t,i] * self.transition_mat[i,j] * self.emission_mat[j, sequence[t+1]] * self.beta[t+1, j]
                    self.gamma[t, i] += self.xi[t,i,j]
        
        self.gamma[T-1] = self.alpha[T-1].copy()

    def e_step(self, sequence):
        '''compute alpha, beta, gamma, xi given 
        transition matrix/emission matrix/prior knowledge'''

        self.forward(sequence)
        self.backward(sequence)
        self.get_gamma_xi(sequence)

    def m_step(self, sequence):
        '''re-estimate transition matrix/emission matrix/prior 
        knowledge given alpha, beta, gamma, xi'''

        T = len(sequence)

        # new estimate for prior knowledge
        self.prior_knowledge = self.gamma[0].copy()

        # new estimate for transition matrix
        for i in range(2):
            denom = 0 
            for t in range(T-1):
                denom += self.gamma[t,i]
            for j in range(2):
                numer = 0 
                for t in range(T-1):
                    numer += self.xi[t,i,j]
                self.transition_mat[i,j] = numer/denom
        
        # new estimate for emission matrix 
        for i in range(2):
            denom = 0 
            for t in range(T):
                denom += self.gamma[t,i]
            
            for o in range(2):
                numer = 0
                for t in range(T):
                    if sequence[t] == o : numer += self.gamma[t,i]
            self.emission_mat[i,o] = numer/denom 

        # row-wise normalization of emission matrix
        self.emission_mat /= self.emission_mat.sum(axis=1, keepdims=True)


    def get_evidence(self):
        log_prob = 0.
        for ct in self.c_arr:
            log_prob += np.log(ct)
        return -log_prob
        
class HMM_MULTI:
    '''
    HMM for multiple sequences
    '''

    def __init__(self):
        self.hmm = HMM_SINGLE()
        self.alpha_list = []
        self.gamma_list = []
        self.xi_list = []
        self.evidence = 0.

        # own parameters
        self.prior_knowledge =  None
        self.transition_mat = None
        self.emission_mat = None

    def e_step(self, multiple_sequence):
        self.evidence = 0.
        self.alpha_list.clear()
        self.gamma_list.clear()
        self.xi_list.clear()
        for seq in multiple_sequence:
            self.hmm.e_step(seq)
            self.alpha_list.append(self.hmm.alpha)
            self.gamma_list.append(self.hmm.gamma)
            self.xi_list.append(self.hmm.xi)
            self.evidence += self.hmm.get_evidence()

    def m_step(self, multiple_sequence): 
        num_seq = len(multiple_sequence)
        self.prior_knowledge = np.zeros((2,), dtype=np.float)
        self.emission_mat = np.zeros((2,2), dtype=np.float)

        numer_t = np.zeros((2,2), dtype=np.float)
        denom_t = np.zeros((2,1), dtype=np.float)
        numer_e = np.zeros((2,2), dtype=np.float)
        denom_e = np.zeros((2,1), dtype=np.float)

        for r in range(num_seq):
            self.prior_knowledge += self.gamma_list[r][0]
            for t, o in enumerate(multiple_sequence[r]):
                numer_e[:,o] += self.gamma_list[r][t]
                denom_e[:,0] += self.gamma_list[r][t]
                if t < len(multiple_sequence[r])-1 : 
                    numer_t += self.xi_list[r][t]
                    denom_t[:,0] += self.gamma_list[r][t]
                
        self.prior_knowledge /= num_seq
        self.transition_mat = numer_t / denom_t
        self.emission_mat = numer_e / denom_e

        # update params for internal hmm
        self.hmm.prior_knowledge = self.prior_knowledge.copy()
        self.hmm.transition_mat = self.transition_mat.copy()
        self.hmm.emission_mat = self.emission_mat.copy()

    def get_evidence(self):
        return self.evidence

class ConvMonitor:
    def __init__(self):
        self.results = [-np.inf]

    def update(self, new_result):        
        stop_update = True if new_result < self.results[-1] else False
        self.results.append(new_result)
        return stop_update 
        
if __name__ == '__main__':
    obs = [[0,1,0,1,0,1,1],
            [0,1,1,0,1,0,1,0,1,1,1],
            [0,0,1,1,1,1,1]]

    model = HMM_MULTI()
    monitor = ConvMonitor()
    
    for it in range(MAX_ITERS):
        model.e_step(obs)
        model.m_step(obs)
        print('ITER {:02d} | loglike : {:06f}'.format(it+1, model.get_evidence()))
        if monitor.update(model.get_evidence()) : break
    print(f"transition matrix : {model.transition_mat}")
    print(f"emission matrix : {model.emission_mat}")
    print(f"initial hidden states :{model.prior_knowledge}")

