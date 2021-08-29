'''
Created on May 11, 2018

@author: dduque
This module generates all the random numbers required to sample.
'''

import numpy as np
'''
Random stream to generate forward passes in the optimization phase
'''
alg_rnd_gen = np.random.RandomState(0)
'''
Random stream to simulate the policy with in sample, i.e., with the data that the model was train with
'''
in_sample_gen = np.random.RandomState(1234)
'''
Random stream to simulate the policy with in sample, i.e., with the data for out-of-sample testing
'''
out_sample_gen = np.random.RandomState(1111)
'''
Random stream to design an instance
'''
experiment_desing_gen = np.random.RandomState(2405)  # 1928  # 2405
seeds = {alg_rnd_gen: 0, in_sample_gen: 1234, out_sample_gen: 1111, experiment_desing_gen: 2405}


def reset_alg_rnd_gen():
    alg_rnd_gen.seed(seeds[alg_rnd_gen])


def reset_in_sample_gen():
    in_sample_gen.seed(seeds[in_sample_gen])


def reset_out_sample_gen():
    out_sample_gen.seed(seeds[out_sample_gen])


def reset_experiment_desing_gen():
    experiment_desing_gen.seed(seeds[experiment_desing_gen])


def reset_all_rnd_gen():
    reset_alg_rnd_gen()
    reset_in_sample_gen()
    reset_out_sample_gen()


if __name__ == '__main__':
    for i in range(10):
        print(alg_rnd_gen.lognormal(1, 1), in_sample_gen.lognormal(1, 1), out_sample_gen.lognormal(1, 1))
    reset_alg_rnd_gen()
    reset_in_sample_gen()
    reset_out_sample_gen()
    print('---------------')
    for i in range(10):
        print(alg_rnd_gen.lognormal(1, 1), in_sample_gen.lognormal(1, 1), out_sample_gen.lognormal(1, 1))
