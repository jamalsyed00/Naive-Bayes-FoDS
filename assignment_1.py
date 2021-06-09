import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy
from scipy.stats import beta

"""DATASET"""

coin_toss = random.choices(range(0,2), weights =[0.31,0.69], k=160)
df = pd.DataFrame(coin_toss)

def max_likelihood_estimator(df):
  m=0
  for i in df[0]:
        if i == 1:
          m = m + 1
  return m/len(df), m, len(df) - m

"""LISA'S METHOD"""

uml , m , l = max_likelihood_estimator(df)
print('Maximum likelihood estimator uml= ', uml)

a , b = 4 , 6 
pm_lisa = (m+a)/(m+a+l+b)
print('Posterior mean(Lisa) = ', pm_lisa)

x = np.arange (0.01, 1, 0.01)
y = beta.pdf(x,a+m,b+l)
plt.plot(x,y)
plt.title('Posterior Distribution (Lisa)')
plt.savefig("Posterior_Lisa.png",bbox_inches='tight',dpi = 150)
plt.show()

"""BOB'S METHOD"""

count = 0
for i in df[0]:
    y = beta.pdf(x,a,b)
    plt.plot(x,y)
    plt.ylim(0,12) 
    plt.title('Posterior Distribution (Bob)')
    plt.savefig('line plot'+str(count)+'.png',bbox_inches='tight', dpi=150)
    count+=1
    if i==1:
        a=a+1
    else:
        b=b+1
    plt.clf()

plt.show()
pm_bob = a/(a+b)
print('Posterior mean(Bob)= ', pm_bob)

y = beta.pdf(x,a,b)
plt.plot(x,y)
plt.ylim(0,12) 
plt.title('Posterior Distribution (Bob) After all iterations')
plt.show()
