# Проект Understanding customer

---
## Постановка задачи

_Цель проекта_ - определение (классификация) намерений (интентов) пользователей по их запросам в общении с чат-ботом или виртуальном ассистентом веломагазина. <br><br>
По размеченному обучающему набору данных тренировочных  диалогов чатбота веломагазина, с помощью различных моделей машинного обучения нужно сделать прогнозы намерений для запросов пользователей из тестового файла.<br><br>
При этом, необходимо: <br>
1. **Машинное обучение**  
  _Для классификации намерений пользователя использовать следующие архитектуры: <br>
     &ensp; A. RNN <br>
     &ensp; B. LSTM <br>
    &ensp;  C. BERT_<br>

2. **Описание проекта**  
    _В ноутбуке следует указать параметры окружения для решений: используемая архитектура, веса, предварительная обработка, количество слоев, фреймворк и т. д., а также метрики обучения и валидации._<br>

3. **Результаты**  
 &ensp; А. _Полученные модели должны быть сохранены в формате pickle_<br>
 &ensp; B. _Результаты для тестового набора должны быть сохранены в файл с именем intents.csv_<br>
 &ensp; C. _Добиться **точности не менее 0,8** на тестовом наборе данных_<br>

4. _Bonus_: <br>
    * _Для классификации намерений пользователя использовать архитектуру CNN_
    * _Для получения лучших результатов расширить тренинговый набор данных, добавив больше различных фраз и соответствующих намерений_
    * _Добиться **точности не менее 0,873** на тестовом наборе данных_
   


## Начало Работы

### Копирование
Для копирования файлов Проекта на локальный компьютер в папку *<your_dir_on_local_computer>* выполните:

```
    $ git clone git@github.com:dbadeev/tweets.git <your_dir_on_local_computer>
```

### Описание файлов
* *tweets.pdf* - текст задания
* *requirements.txt* - список библиотек, необходимый для работы
* Папка *data*
  - *processedNegative.csv* - файл с твитами негативной тональности
  - *processedNeutral.csv*  - файл с твитами нейтральной тональности
  - *processedPositive.csv*  - файл с твитами позитивной тональности
* *tweets.ipynb* - ноутбук проекта  
* *text_cleaninig.py* - утилиты "чистки" текста твитов
* *text_processing.py* - утилиты векторизации текста твитов
* *w2v_ml.py* - утилиты векторизации с помощью Word2Vec и предварительно обученных моделей представления векторов слов
* *cosine_similarity.py* - утилиты вычисления косинусного сходства векторов представлений слов
* *machine_learning.py* - утилиты нахождения оптимальных параметров различных моделей машинного обучения для подсчета accuracy с помощью GridSearch
* Папка *res*
  - *cos_sim.csv* - файл с 10 наиболее схожими парами твитов среди датасетов, полученных в результате использования различных подходов к предварительной обработке данных 
  - *df_df_prep.csv*  - файл с твитами, полученными в результате использования различных подходов к предварительной обработке данных 
<br>

## Запуск
В файле *tweets.ipynb* приведена пошаговая реализация проекта с пояснениями и промежуточными результатами. 

## Авторы

*loram (Дмитрий Бадеев)* <br>
*gdorcas (Татьяна Смирнова)*

<br><br>

## Результат в School 21
<img width="640" alt="image" src="https://github.com/user-attachments/assets/1f4ce7e5-20a9-4fe2-8480-30b83ea938c0">
