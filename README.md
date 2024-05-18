| Название исследуемой задачи | Применение стохастической аппроксимации нулевого порядка с техникой запоминания в алгоритме Франка-Вульфа |
| :---: | :---: |
| Тип научной работы | Выпускная квалификационная работа |
| Автор | Богданов Александр Иванович |
| Научный руководитель | к.ф.-м.н. Безносиков Александр Николаевич |

# Описание

В данной работе рассматривается проблема оптимизации "черного ящика". В такой постановке задачи не имеется доступа к градиенту целевой функции, поэтому его необходимо как-то оценить. Предлагается новый способ аппроксимации $\texttt{JAGUAR}$, который запоминает информацию из предыдущих итераций и требует $\mathcal{O}(1)$ обращений к оракулу. Я реализую эту аппроксимацию для алгоритма Франка-Вольфа и докажу сходимость для выпуклой постановки задачи. Также в данной работе рассматривается стохастическая задача минимизации на множестве $Q$ с шумом в оракуле нулевого порядка, такая постановка довольно непопулярна в литературе, но мы доказали, что $\texttt{JAGUAR}$-аппроксимация является робастной не только в детерминированных задачах минимизации, но и в стохастическом случае. Я провел эксперименты по сравнению моего градиентного оценщика с уже известными в литературе и подтверждаю доминирование своего метода.

# Установка

- Second way
1. `git clone` this repository.
2. Create new `conda` environment and activate it
3. Run 
```bash
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name <env_name> --display-name <env_name>
```

# Содержание

This repository provides code for reproducing experiments that were performed as part of scientific work in the fall semester of 2023. If you run [Experiments.ipynb](https://github.com/intsystems/Bogdanov-BS-Thesis/blob/main/code/Experiments.ipynb) in the code directory, you will reproduce the experimental results obtained in the article. 

![JAGUAR](./code/figures/Non-stochastics.png)

