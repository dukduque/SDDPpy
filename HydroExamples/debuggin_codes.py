'''
Created on Oct 18, 2018

@author: dduque
'''
import numpy as np
t=0
sp0 = algo.stage_problems[t]
rnd_container = algo.random_container
desc_rnd_vec = rnd_container[t+1]
p_w = np.zeros(desc_rnd_vec.outcomes_dim)
for (i,j) in sp0.risk_measure.dro_ctrs:
    p_w[j]  += sp0.risk_measure.dro_ctrs[(i,j)].Pi

print(p_w)

r1 = 0
r2 = 2

x = desc_rnd_vec.elements['innovations[%i]' %r1].outcomes
print(x[p_w>0])
y = desc_rnd_vec.elements['innovations[%i]' %r2].outcomes
print(y[p_w>0])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(x):
    ax.annotate("(%.2f,%.2f)" %(x[i], y[i]), (x[i], y[i]))

x = rnd_container_data[t+1].elements['innovations[%i]' %r1].outcomes
y = rnd_container_data[t+1].elements['innovations[%i]' %r2].outcomes
ax.scatter(x, y)