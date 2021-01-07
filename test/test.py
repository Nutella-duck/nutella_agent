'''
    nutella로 hpo 돌린 결과를 가지고 fanova 사용해보기
    hp들을 dataframe에 담고, loss를 response로 만들기
    기존 tutorial에서 space를 더 쉽게 만든 버전
'''

from nutellaAgent import hpo
from nutellaAgent import nu_fmin
import numpy as np
import pandas as pd

# define an objective function
def objective(args):
    val, val2 = args['hp1'], args['hp2']
    if val>val2:
        return val + val2
    else:
        return val ** 2 - val2

# define a search space
# space = hpo.hp.choice('a', [('case 1', 1 + hpo.hp.lognormal('c1', 0, 1)),
#                             ('case 2', hpo.hp.uniform('c2', -10, 10))])
space = {'hp1': 1 + hpo.hp.lognormal('a', 0, 1),
         'hp2': hpo.hp.uniform('b', 1, 3)
        }

# minimize the objective over the space
trials = hpo.Trials()
best = nu_fmin(objective, space, algo=hpo.tpe.suggest, max_evals=100, trials=trials)

print(best)

# print(space.keys())
# print(type(space['hp1']))

# hp들을 x로
hps = trials.vals
hps_list = list(hps.keys())

features = []

for i in range(len(trials.tids)):
    tmp =[]
    for j in range(len(hps_list)):
        tmp.append(hps[hps_list[j]][i])
    features.append(tmp)

columns_list = []
for i in range(len(hps_list)):
    columns_list.append(str(i))

df_x = pd.DataFrame(features, columns=columns_list)

# loss를 y로
res = trials.results

responses = []
for i in range(len(res)):
    responses.append(res[i]['loss'])
responses = np.array(responses)

# print(res[0]['loss'])
# print(trials.results)
