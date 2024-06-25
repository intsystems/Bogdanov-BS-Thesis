# Руководство

Данная директория содержит код для выпускной квалификационной работы бакалавра "Аппроксимации градиента с помощью оракула нулевого порядка и техники запоминания".

## Содержание
1. [Установка](#installation)
2. [Использование](#usage)
3. [Описание файлов](#files-description)

## Установка <a name="installation"></a>
Чтобы использовать этот проект, вам необходимо иметь Python и Jupyter Notebook установленные на вашем компьютере. Вы можете скачать Python [отсюда](https://www.python.org/downloads/) и Jupyter Notebook [отсюда](https://jupyter.org/install).

Склонируйте репозиторий:
```bash
git clone https://github.com/intsystems/Bogdanov-BS-Thesis.git
```

Перейдите к директорию с кодом:
```bash
cd code
cd experiments
```

Установите необходимые пакеты:
```bash
pip install -r requirements.txt
```

## Использование <a name="usage"></a>
Чтобы запустить проект, откройте Jupyter Notebook:
```bash
jupyter notebook
```
Далее, откройте файл `L1.ipynb`, `L2.ipynb` или 'Simplex.ipynb' (зависит от множества, на котором будут воспроизводиться эксперименты) в интерфейсе Jupyter Notebook.

## Описание файлов в `files` <a name="files-description"></a>

- `run_experiments.py`: содержит инициализацию эксперимента.
- `optimizers.py`: содержит классы оптимизаторов.
- `gradient_approximation.py`: содержит классы аппроксиматоров.
- `sets.py`: cодержит классы множеств.
- `utils.py`: cодержит вспомогательные функции.
