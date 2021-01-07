![PyPI Depend](https://img.shields.io/badge/PyPI-v0.1.20-orange) ![Python Depend](https://img.shields.io/badge/Python-3.7-blue) ![License Badge](https://img.shields.io/badge/license-MIT-green)<br>

<p align="center">
  <img width="400" src="./assets/logo.jpeg">
</p>


<h2 align=center>NutellaAgent: Hyperparameter Optimization and Visualization of ML Metrics</h2>

[NutellaAgent](https://github.com/NutellaCoder/NutellaML)는 Nutella web 서비스를 위한 파이썬 라이브러리이다. 이 라이브러리를 통해 사용자는 다양한 머신러닝 실험 지표들을 시각화 할 수 있으며, 자신의 모델에 최적화된 하이퍼파라미터를 구할 수 있다. 또한, 하이퍼파라미터를 최적화한 결과와 각 파라미터들이 결과에 얼마나 중요한 역할을 하는지 또한 볼 수 있다. 이 모든 것은 Netella Web에서 손쉽게 확인할 수 있다.


## Installation

```bash
$ pip install nutellaAgent
```

## Visualization

Netella Web에서 로그인을 하고 프로젝트를 생성한 뒤에 어디서든 자신의 모델을 실행시켜 원하는 지표를 시각화할 수 있다.

```python
import nutellaAgent

# 첫 번째 모델 객체 생성
first_model = nutellaAgent.Nutella()

# 모델 이름 설정 및 프로젝트 연결, 웹 페이지에서 생성한 프로젝트의 키
first_model.init("model_name", "project_api_key", 0)

# 모델 학습
...

# 시각화하고 싶은 지표 전송
first_model.log(accuracy = acc, loss = loss)
```


## Hyperparameter Optimization

Netella Web에서 HPO 프로젝트를 생성한 뒤에 최적화된 하이퍼파라미터 값을 얻을 수 있으며, 웹 페이지를 통해 각 하이퍼파라미터들이 output에 영향을 미치는 정도를 한눈에 파악할 수 있다.

```python
from nutellaAgent import hpo, nu_fmin

# objective function 정의
def objective(args):
    val, val2 = args['hp1'], args['hp2']
    if val > val2:
        return val + val2
    else:
        return val ** 2 - val2

# search space 정의
space = {'hp1': 1 + hpo.hp.lognormal('a', 0, 1),
         'hp2': hpo.hp.uniform('b', 1, 3)}

# hpo 실행
trials = hpo.Trials()
best = nu_fmin(objective, space, algo=hpo.tpe.suggest, max_evals=100, trials=trials)

# 최적의 hyperparameter 값 출력
print(best)
```

## More Examples

[tutorials](https://github.com/NutellaCoder/NutellaML/tree/master/tutorials)에서 확인할 수 있다.


## License

Copyright © [NutellaCoder](https://github.com/NutellaCoder)

[nutellaAgent](https://github.com/NutellaCoder/NutellaML) is open-sourced software licensed under the [MIT License](https://github.com/NutellaCoder/NutellaML/blob/master/LICENSE).