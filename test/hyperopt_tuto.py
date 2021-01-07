# Hyperopt 튜토리얼

from hyperopt import hp, fmin, tpe, space_eval, Trials
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

print(best)
# print(trials.results)
# -> {'a': 1, 'c2': 0.01420615366247227}
# print(space_eval(space, best))
# # -> ('case 2', 0.01420615366247227}

################ JSON 테스트
import json

all_info_dict = dict()
# all_info_dict["method"] = tpe.suggest
# all_info_dict["best_result"] = trials.best_trial['result'] 

tmp = {'a': 1, 'c2': 0.03511167342948704}

all_info_dict["best_hp"] = tmp
# all_info_dict["trial_result"] = trials.results 
# all_info_dict["trial_hp"] = trials.vals
# print(type(all_info_dict))
all_info = json.dumps(all_info_dict)

print(all_info)