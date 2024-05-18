| Название исследуемой задачи | Применение стохастической аппроксимации нулевого порядка с техникой запоминания в алгоритме Франка-Вульфа |
| :---: | :---: |
| Тип научной работы | Выпускная квалификационная работа |
| Автор | Богданов Александр Иванович |
| Научный руководитель | к.ф.-м.н. Безносиков Александр Николаевич |

# Описание

В данной работе рассматривается проблема оптимизации "черного ящика". В такой постановке задачи не имеется доступа к градиенту целевой функции, поэтому его необходимо как-то оценить. Предлагается новый способ аппроксимации $\texttt{JAGUAR}$, который запоминает информацию из предыдущих итераций и требует $\mathcal{O}(1)$ обращений к оракулу. Я реализую эту аппроксимацию для алгоритма Франка-Вольфа и докажу сходимость для выпуклой постановки задачи. Также в данной работе рассматривается стохастическая задача минимизации на множестве $Q$ с шумом в оракуле нулевого порядка, такая постановка довольно непопулярна в литературе, но мы доказали, что $\texttt{JAGUAR}$-аппроксимация является робастной не только в детерминированных задачах минимизации, но и в стохастическом случае. Я провел эксперименты по сравнению моего градиентного оценщика с уже известными в литературе и подтверждаю доминирование своего метода.

# Установка

1. `git clone` этот репозиторий
2. Создайте новое окружение `conda` и активируйте его
3. Запустите 
```bash
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name <env_name> --display-name <env_name>
```

# Содержание

В этом репозитории представлен код написанный в рамках выпускной квалификационной работы. Если вы запустите [L1.ipynb](https://github.com/intsystems/Bogdanov-BS-Thesis/blob/main/code/L1.ipynb), [L2.ipynb](https://github.com/intsystems/Bogdanov-BS-Thesis/blob/main/code/L2.ipynb) или [Simplex.ipynb](https://github.com/intsystems/Bogdanov-BS-Thesis/blob/main/code/Simplex.ipynb) в разделе code, то вы воспроизведете экспериментальные результаты, полученные в работе.  

![JAGUAR](./code/figures/Non_stochastics_FW_LogReg_Simplex.png)
![JAGUAR](./code/figures/Non_stochastics_FW_LogReg_L2.png)
![JAGUAR](./code/figures/Non_stochastics_FW_Reg_Simplex.png)
![JAGUAR](./code/figures/Stochastics_TPF_FW_Reg_Simplex.png)
