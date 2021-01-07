'''
    nutella로 hpo 돌린 결과를 가지고 fanova 사용해보기
    hp들을 dataframe에 담고, loss를 response로 만들기
'''

from nutellaAgent import hpo
from nutellaAgent import nu_fmin
import numpy as np
import pandas as pd

# define an objective function
def objective(args):
    case, val = args['hp1']
    val2 = args['hp2']
    if case == 'case 1':
        return val + val2
    else:
        return val ** 2 - val2

# define a search space
# space = hpo.hp.choice('a', [('case 1', 1 + hpo.hp.lognormal('c1', 0, 1)),
#                             ('case 2', hpo.hp.uniform('c2', -10, 10))])
space = {'hp1': hpo.hp.choice('a', 1 + hpo.hp.lognormal('c1', 0, 1)),
                                    ('case 2', hpo.hp.uniform('c2', -10, 10))]),
         'hp2': hpo.hp.uniform('b', 1, 3)
        }


# minimize the objective over the space
trials = hpo.Trials()
best = nu_fmin(objective, space, algo=hpo.tpe.suggest, max_evals=100, trials=trials)

print(best)

print(space.keys())
# print(type(space['hp1']))

# hp들을 x로
hps = trials.vals
# print(hps['a'][0])

features = []
idx_of_c1 = 0
idx_of_c2 = 0

list_hp = list(space.keys())
print(list_hp[0])
# print(hps[list_hp[0]])

for i in range(len(hps['a'])):
    if hps['a'][i] == 0:
        features.append([hps['c1'][idx_of_c1], hps['b'][i]])
        idx_of_c1 += 1
    else:
        features.append([hps['c2'][idx_of_c2], hps['b'][i]])
        idx_of_c2 += 1

df_x = pd.DataFrame(features, columns=['0','1'])


# loss를 y로
res = trials.results

responses = []
for i in range(len(res)):
    responses.append(res[i]['loss'])
responses = np.array(responses)

# print(res[0]['loss'])
# print(trials.results)
