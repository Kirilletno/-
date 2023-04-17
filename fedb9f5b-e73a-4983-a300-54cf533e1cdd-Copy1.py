#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px">
# <b>Кирилл, привет!</b>
# 
# Меня зовут Андреева Юлия, я буду ревьюером на этом проекте.
# 
# Твой проект легко читать и по нему удобно перемещаться — есть структура, пояснения, печать результатов моделей и выводы. Почти все ключевые этапы в работе выполнены.
# 
# Есть важные моменты, которые нужно будет исправить и дополнить. Они касаются неинформативных признаков, кодировки признаков, стандартизации, методов борьбы с дисбалансом. Всё это подробней описано по ходу работы и дано общим списком в итоговом комментарии.
#     
# Я буду оформлять свои комментарии следующим образом:
# 
# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Так выделены важные замечания. Без их отработки проект не будет принят. </div>
# 
# <div class="alert alert-warning">
# <b>Комментарий ревьюера ⚠️:</b> Так выделены небольшие замечания. Их можно учесть при выполнении будущих заданий или доработать проект сейчас (однако это не обязательно).</div>
# 
# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Так я выделяю все остальные комментарии.</div>
#     
# Пример оформления для твоих комментариев:
# <div class="alert alert-info"> <b>Комментарий студента:</b> Например, вот так.</div>
#     
# Если у тебя есть какие-то вопросы по проекту — задавай, я постараюсь всё разъяснить 😉

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# ## Подготовка данных

# Импортируем библиотеки, которые понадобятся нам в ходе работы

# In[2]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder


# <div class="alert alert-warning">
# <b>Комментарий ревьюера ⚠️:</b> Обращай внимание на то, чтобы импортировались только те библиотеки, пакеты, которые будут использоваться в текущем проекте (OrdinalEncoder лишний). Иногда импорт лишних библиотек может приводить к большим временным затратам и потреблению ресурсов памяти.

# <div class="alert alert-info"> <b>Комментарий студента:</b> Спасибо, исправил, не знал об этом. </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> 👍😉
# </div>

# Изучаем полученные данные

# In[3]:


df = pd.read_csv('/datasets/Churn.csv')
df.info()
df.head()


# Подготавливаем данные для последующего анализа, преобразуем категориальные признаки в колличественные методом OHE, а также заполняем пропуски и бесконечности из таблицы, во избежангие появления ошибки- ValueError: Input contains NaN, infinity or a value too large for dtype('float64'). в дальнейшем, при тренировки модели (происходила из-за пропусков в столбце Tenure)
# 
# Также, чтобы не плодить огромное колличество столбцов и не увеличивать время работы модели, после преобразования колличественных переменных в категориальные, удаляем столбец- Surname

# df = df.drop('Surname', axis=1)
# df_ohe = pd.get_dummies(df, drop_first=True)
# df_ohe = df_ohe.replace((np.inf, np.nan), 0).reset_index(drop=True)

# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Удалены не все неинформативные признаки.
#     
# Также хочу обратить внимание на важный момент. Категориальные признаки нужно кодировать после разделения на выборки. При обучении моделей мы пытаемся воссоздать реальную ситуацию, когда у нас имеется только текущие данные (тренировочная + валидационная выборка), а тестовой выборки нет, т.к. это «данные из будущего». Поэтому мы знаем только какие категории есть в наших текущих данных (в <i>geography</i> это France, Spain, Germany, а в <i>gender</i> — Male, Female). А в «будущих данных» может появиться неизвестная ранее категория, которой не было изначально (например, если появится еще одна страна Italy в столбце <i>geography</i>). В таком случае <code>get_dummies</code> уже не справится, появится ошибка. А <code>OneHotEncoder()</code> с такими ситуациями справляется.
#     
# В данном проекте все категории одинаковы, что в тренировочной, что в тестовой выборке, поэтому ты можешь использовать get_dummies, но только нужно это сделать после разделения на выборки для каждой выборки. После этого нужно сделать проверку, что получились одни и те же фичи в каждой выборке (с одинаковыми названиями и одинаковое количество этих колонок с закодированными значениями).
# </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Правильно, что используешь аргумент drop_first=True — после кодирования не должно быть избыточных переменных. В технике OHE его также надо использовать.
# </div>

# <div class="alert alert-warning">
# <b>Комментарий ревьюера ⚠️:</b> В будущих проектах будет требоваться fit OHE только на тренировочной выборке (и transform на остальные выборки). Ты можешь попробовать сделать этот вариант кодировки сейчас. В этом проекте это не обязательно, а в следующих спринтах это уже обязательное требование.
#     
# get_dummies подходит для анализа данных, а для машинного обучения более предпочтителен OHE, т.к. он позоволяет избежать ряд ошибок при обучении моделей (одну из них я описала в комментарии выше).
#     
# Скидываю статью с наглядным примером о разнице OneHotEncoder() vs get_dummies:
# https://albertum.medium.com/preprocessing-onehotencoder-vs-pandas-get-dummies-3de1f3d77dcc
#     
# Про разницу между Ordinal Encoding и One-Hot-Encoding: https://stackoverflow.com/questions/69052776/ordinal-encoding-or-one-hot-encoding
#     
# Почему нужно делать OHE после разделения на выборки с примером: https://stackoverflow.com/questions/55525195/do-i-have-to-do-one-hot-encoding-separately-for-train-and-test-dataset
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Понял, благодарю. Внес исправление ниже + добавлю, предыдущий вывод отменил.
# Удаляем неинформативные признаки: Surname, RowNumber, CustomerId. + Заполняем пропуски в столбце Tenure и других пропусках датасета на ноль </div> 

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> Хорошо!
# </div>

# In[4]:


df = df.drop(['Surname', 'CustomerId', 'RowNumber'], axis=1)
df = df.replace((np.inf, np.nan), 0).reset_index(drop=True)


# ВОПРОС РЕВЬЮЭРУ- пропуски и бесонечности в итоге, было решено заполнить нудевым значением, а не медианой, чтобы не искажать итоговые данные, так как мы не знаем причин пропусков, насколько это верное решение?

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Чтобы не искажать распределение данных пропуски заменяют на медиану. Или можно делать замену на другие значения, аргументировав это решение. В данном случае 0 в tenure может означать, что человек был клиентом банка меньше года, поэтому такая замена допустима.
# </div>

# Выделяем целевой признак, а также делим выборки на валидационную, тренировочную и тестовую для последующей работы 

# target = df['Exited']
# features = df.drop('Exited', axis=1)
# 
# features_train, features_test1, target_train, target_test1 = train_test_split(
#     features, target, test_size=0.4, random_state=12345)
# features_valid, features_test,target_valid, target_test = train_test_split(
#     features_test1, target_test1, test_size=0.5, random_state=12345)
# 
# display(features_train.dtypes)

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Формирование выборок прошло успешно. Хорошо, что не забываешь указывать <code>random_state</code> — это позволит при необходимости воспроизвести такое же разделение на выборки.
# </div>

# <div class="alert alert-warning">
# <b>Комментарий ревьюера ⚠️:</b> При делении на выборки желательно задавать параметр stratify, потому что в противном случае метрика попадает в зависимость от псевдослучайных значений (в зависимости от random_state, например, в трейн может попасть мало значений класса 1, отсюда низкие предсказывания этого класса на других выборках).
#     
# Также стоит печатать размерность получившихся выборок для наглядности и это может помочь вовремя заметить ошибки при разделении.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Спасибо, закоментировал предыдущий вывод, прикрепляю ниже верный вариант с учетом параметра stratify и выводом подсчета значений в каждой выборке. </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> Отлично!
# </div>

# In[5]:


target = df['Exited']
features = df.drop('Exited', axis=1)

features_train, features_test1, target_train, target_test1 = train_test_split(
    features, target, test_size=0.4, random_state=12345,  stratify= target)
features_valid, features_test,target_valid, target_test = train_test_split(
    features_test1, target_test1, test_size=0.5, random_state=12345, stratify= target_test1)

print(f"Количество строк в target_train по классам: {np.bincount(target_train)}")
print(f"Количество строк в target_valid по классам: {np.bincount(target_valid)}")
print(f"Количество строк в target_test по классам: {np.bincount(target_test)}")


# <div class="alert alert-info"> <b>Комментарий студента:</b> Также, провожу раздельное преобразование каждой выборки с помощью OHE, отдельно для каждой выборки, после их разделения.
#     
# P.S. Не совсем понял, как в итоге выполнять кодирование с помощью техники OHE, вместо pd.get dummies, нашел только такую статью- https://www.codecamp.ru/blog/one-hot-encoding-in-python/ Но кажется что должен быть более легкий способ проводить его, или тут описан единственный верный алгоритм? Насколько я понял, OHE работает только со строками и пока мне не совсем понятно, как это можно обойти   

# <div class="alert alert-warning">
# <b>Комментарий ревьюера v2⚠️: </b> Да, в статье показан один из вариантов реализации. Для OHE из библиотеки preprocessing нужно указать параметры <code>handle_unknown</code> и <code>drop='first'</code>. Первый отвечает за то, как обрабатывать неизвестные ранее категории. По умолчанию он равен 'error' — появляется ошибка, если во время transform появляется неизвестная категория (как и при get_dummies). А если поставить этот параметр равным 'ignore', тогда при встрече неизвестной категории, в столбцах с закодированными имеющимися категориями будут стоять нули. 
#     
# Далее можно сделать разными способами. Один из них — после создания энкодера и fit, в конце ещё нужно добавить toarray() <code>encoder.fit_transform(features_train[names]).toarray()</code>, чтобы кодировка отобразилась в виде массивов и можно было из них создать датафрейм. И затем для каждой выборки тебе нужно будет соединить 2 датафрайма — с закодированными категориальными переменными и остальными.
#     
# Пример кода для такого случая:
# 
#     encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
# 
#     features_train_ohe = pd.DataFrame(
#         encoder.fit_transform(features_train[var_categorical]).toarray(),
#         columns=encoder.get_feature_names_out()
#     )
#     
#     scaler = StandardScaler()
# 
#     features_train_numeric = pd.DataFrame(
#         scaler.fit_transform(features_train[var_numeric]),
#         columns=var_numeric
#     )
#     
#     features_train_enc = features_train_ohe.join(features_train_numeric)
#     
# В этом блокноте показан другой вариант кодирования с помощью OHE, а также есть объяснения make_column_transformer и make_pipeline с примером. 
#     
# https://colab.research.google.com/drive/1_gAMXcQKoCShB_l8FNtYEejMnosm9mvt?usp=sharing
# </div>

# In[6]:


features_train = pd.get_dummies(features_train, drop_first=True)
features_valid = pd.get_dummies(features_valid, drop_first=True)
features_test = pd.get_dummies(features_test, drop_first=True)
display('Столбцы и значения  Датафрейма features_train', features_train.info())
display('Столбцы и значения Датафрейма features_valid', features_valid.info())
display('Столбцы и значения  Датафрейма features_test', features_test.info())


# Приводим все данные к единому масштабу, во избежание ситуаций, в которых система, может воспринять какое-либо из значений более важным, нежели другие

# In[7]:


pd.options.mode.chained_assignment = None

numeric = ['CreditScore', 'Balance', 'Age', 'Tenure', 'EstimatedSalary' ]

scaler = StandardScaler()
scaler.fit(features_train[numeric]) 

features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])

features_train


# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Не забывай делать transform на все выборки. Пропустил тестовую.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Спасибо, пропустил этот момент. Исправаил ниже </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> 👍
# </div>

# In[8]:


features_test[numeric] = scaler.transform(features_test[numeric])


# ВОПРОС РЕВЬЮЭРУ- Насколько при обучении модели нам важны строки с фамилией, порядковым номером и CustomerId или их можно выкинуть на этапе обработки, чтобы не смущать модель и не затруднять ей работу, а также ликвидировать необходимость поиска закономерностей на это этапе или эти данные могут понадобиться нам позже?

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Эти признаки важной информации не несут, они нужны только для базы данных по клиентам (хранение и обращение к данным). Если включать эти признаки в модель машинного обучения, модель будет пытаться найти закономерности на основе этих признаков и строить прогнозы с их учётом, когда на самом деле никаких закономерностей нет, это будет замедлять работу и ухудшать качество модели. Неинформативные признаки удаляются на этапе подготовки данных.
# </div>

# ## Исследование задачи

# Исследуем баланс классов, проверяем насколько 0 преобладают над 1 в полученной выборке

# In[9]:


target_zeros = target[target == 0]
target_ones = target[target == 1]
print(target_zeros.shape)
target_ones.shape


# Разница довольно ощутима, дисбаланс на лицо
# Однако, сперва обучим разные модели без учета дисбаланса классов и сравним результаты, с работой модели, на более сбалансированных данных

# Сперва проверим модель логистическое регрессии

# In[10]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train) 
predicted_valid = model.predict(features_valid)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)


print("F1:", f1_score(target_valid, predicted_valid))
print('AUC-ROC - ', auc_roc)


# Протестируем также модель дерева решений, чтобы найти наилучшую метрику f1, протестировав разные варианты глубины

# In[11]:


best_result = 0
best_depth = 0
best_model = None

for depth in range(1, 20):
   
    model = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    result = f1_score(target_valid, predicted_valid)
    
    if result > best_result:
        best_result = result
        best_depth = depth
        best_model = model
        
probabilities_valid = best_model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)        
        
        
print('Лучшая глубина-', best_depth, 'и результат-', best_result)
print('AUC-ROC - ', auc_roc)


# Финально протестируем также и модель случайного леса, перебирая разную глубину и колличество решающих деревьев для поиска наилучшего результата

# In[12]:


best_result = 0
best_est = 0
best_depth = 0

for est in range(10, 51, 10):
    for depth in range (1, 30):
        model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth) 
        model.fit(features_train, target_train)
        predictions_valid = model.predict(features_valid) 
        result = f1_score(target_valid, predictions_valid) 
        if result > best_result:
            best_result = result
            best_est = est
            best_depth = depth
            best_model = model
        
probabilities_valid = best_model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)  
            
print('Лучшая глубина-', best_depth, ', лучшее количество деревьев- ', best_est, 'и лучший результат-', best_result)  
print('AUC-ROC - ', auc_roc)


# Вывод:
# - В самом начале, в классах, был выявлен сильный дисбаланс с преобладанием значений - 0, над 1 в целевом признаке, почти вдвое
# - При работе с несбалансированными классами, метрику F1 не удалось поднять более 0.5 на всх проверенных моделях, несмотря на перебор гиперпараметров
# - Наихудший результат в исследовании дала модель Логистической регрессии с результатом f1, равному 0.3106457242582897, а наилучший дерево решений, при указании параметров глубины в 9 и результата метрики f1 в 0.5843971631205673
# - AUC-ROC растет вместе с F1 и показывает, наилбольшее значение на модели случайного леса
# - Интересно, что случайный лес, дал почти одинаковый результат на несбаланированной выборке, в сравнение с деревом решений, однако в то же время, метрика AUC-ROC случайного леса, получилась наибольшей, что потенциально делает ее более эффективной моделью

# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> «с преобладанием значений - 0, над 1 в целевом признаке, почти вдвое» — не верно, посмотри внимательно.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента: </b> Да, теперь вижу, что разница более ощутима, почти в 4 раза, имелось в виду эта ошибка в формулировке или что-то другое? </div>

# <div class="alert alert-warning">
# <b>Комментарий ревьюера v2⚠️: </b> Все выводы должны соответствовать результатам анализа и моделей. Кроме того, это значение используется в методах балансировки классов. Параметр repeat у тебя был подобран верно, а fraction — нет.
# </div>

# ВОПРОС РЕВЪЮЭРУ Не до конца понимаю, прикладную разницу между метриками f1 и AUC-ROC в чем она заключается? Именно с практической точки зрения
# КОгда лучше использовать AUC-ROC, а когда F1? И что означает сильная разница между их результатами при тесте модели?

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> F1 это среднее гармоническое между precision и recall, а AUC-ROC — площадь под кривой ROC, она сравнивает True Positive Rate с False Positive Rate. Чем больше AUC-ROC, тем больше разница между истинно положительными и истинно отрицательными классами. Если в данных дисбаланс классов, оценка AUC-ROC будет искажена и в таком случае стоит ориентироваться на f1. На практике чаще всего используется f1, т.к. она обладает лучшей интерпретируемостью, а AUC-ROC рассчитывается как дополнительная метрика.
#     
# Статья о том, какие метрики бывают и как их использовать:
#     
# - https://neptune.ai/blog/evaluation-metrics-binary-classification
# </div>

# ## Борьба с дисбалансом

# Чтобы решить проблему дисбаланса, мы используем 2 метода, увеличим колличество редких классов- 1, а также уменьшим колличество классов- 0

# Исследуем начальные значения

# In[13]:


print(features.shape)
print(target.shape)


# Сперва используем технику увеличения классов, со значеним- 1 в выборке 

# In[14]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled = shuffle(features_upsampled, random_state=12345)
    target_upsampled = shuffle(target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled 

features_upsampled, target_upsampled = upsample(features_train, target_train, 4)

print(features_upsampled.shape)
print(target_upsampled.shape)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Параметр <code>repeat</code> определён верно.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента: </b> Добавил нижн, проверку на увеличенной выборке, и далее тоже самое, но на уменьшенной </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> Отлично! Параметр repeat отвечает за то, во сколько раз нам надо увеличить число элементов меньшего класса, чтобы их стало столько же, сколько элементов большего класса.
# </div>

# In[15]:


target_zeros = target_upsampled[target == 0]
target_ones = target_upsampled[target == 1]
print('Нулей в новой выборке- ', target_zeros.shape)
print('Единиц в новой выборке- ', target_ones.shape)


# Отлично, repeat был выбран правильно и кол-во символов сравнялось, теперь проведем тесты на выборке с увеличеной выборкой, для выявления лушчей модели.

# 1. Модель Логистической Регрессии

# In[16]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_upsampled, target_upsampled) 
predicted_valid = model.predict(features_valid)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)


print("F1:", f1_score(target_valid, predicted_valid))
print('AUC-ROC - ', auc_roc)


# 2. Модель дерева решений

# In[17]:


best_result = 0
best_depth = 0
best_model = None

for depth in range(1, 30):
   
    model = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model.fit(features_upsampled, target_upsampled) 
    predicted_valid = model.predict(features_valid)
    result = f1_score(target_valid, predicted_valid)
    
    if result > best_result:
        best_result = result
        best_depth = depth
        best_model = model
        
probabilities_valid = best_model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)        
        
        
print('Лучшая глубина-', best_depth, 'и результат-', best_result)
print('AUC-ROC - ', auc_roc)


# 3. Модель случайного леса

# In[18]:


best_model = None
best_result = 0
best_est = 0
best_depth = 0

for est in range(10, 80, 10):
    for depth in range (1, 30):
        model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth) 
        model.fit(features_upsampled, target_upsampled)  
        predictions_valid = model.predict(features_valid)
        result = f1_score(target_valid, predictions_valid)
        if result > best_result:
            best_model = model
            best_result = result
            best_est = est
            best_depth = depth

        
probabilities_valid = best_model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)  
            
print('Лучшая глубина-', best_depth, ', лучшее количество деревьев- ', best_est, 'и лучший результат-', best_result)  
print('AUC-ROC - ', auc_roc)


# Промежуточный вывод:
# По итогу исследования на выборке, где проблема дисбаланса была решена за счет увеличения выборки, лучший результат, показала модель случайного леса, со значением, метрики f1 - 0.6446469248291572 и АUC-ROC -  0.8716837865799585, интересно как на изменение результата повлияет борьба с дисбалансом через снижение выборки

# Применим, технику уменьшения классов, с уменьшением значения- 0 в выборке, для борьбы с дисбалансом

# In[19]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    features_downsampled = shuffle(pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones]), random_state=12345)
    target_downsampled = shuffle(pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones]), random_state=12345)
    
   
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.26)

print(features_downsampled.shape)
print(target_downsampled.shape)


# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> В <code>downsample</code> указываешь параметр <code>fraction</code> равный 0.96, что неверно. Этот параметр отвечает за то, какую долю класса 0 мы должны взять, чтобы его стало столько же, сколько класса 1.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента: </b> Понял благодарю за объяснение, изначально неверно понял логику подбора гиперпараметра. </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> Прекрасно!😉
# </div>

# In[20]:


target_zeros = target_downsampled[target == 0]
target_ones = target_downsampled[target == 1]
print('Нулей в новой выборке- ', target_zeros.shape)
print('Единиц в новой выборке- ', target_ones.shape)


# ВОПРОС РЕВЬЭРУ- Как понять, насколько лучше увеличить или соответственно уменьшить выборку, нужно чтобы размеры выборки были примерно равын или же нет?

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Да, классы должны быть примерно равны. Понять на сколько можно из отношения классов в первичных данных.
# </div>

# Теперь, когда парамерт fraction был выбран верно, проведем обучение моделей на новых данных, с учетом нового метода борьбы с дисбалансом и сравним результаты, для подбора наилучшей модели.

# 1. Модель Логистической Регрессии

# In[21]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_downsampled, target_downsampled) 
predicted_valid = model.predict(features_valid)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)


print("F1:", f1_score(target_valid, predicted_valid))
print('AUC-ROC - ', auc_roc)


# 2. Модель дерева решений

# In[22]:


best_result = 0
best_depth = 0
best_model = None

for depth in range(1, 30):
   
    model = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model.fit(features_downsampled, target_downsampled) 
    predicted_valid = model.predict(features_valid)
    result = f1_score(target_valid, predicted_valid)
    
    if result > best_result:
        best_result = result
        best_depth = depth
        best_model = model
        
probabilities_valid = best_model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)        
        
        
print('Лучшая глубина-', best_depth, 'и результат-', best_result)
print('AUC-ROC - ', auc_roc)


# 3. Модель случайного леса

# In[23]:


best_model = None
best_result = 0
best_est = 0
best_depth = 0

for est in range(10, 80, 10):
    for depth in range (1, 30):
        model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth) 
        model.fit(features_downsampled, target_downsampled) 
        predictions_valid = model.predict(features_valid)
        result = f1_score(target_valid, predictions_valid)
        if result > best_result:
            best_model = model
            best_result = result
            best_est = est
            best_depth = depth

        
probabilities_valid = best_model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)  
            
print('Лучшая глубина-', best_depth, ', лучшее количество деревьев- ', best_est, 'и лучший результат-', best_result)  
print('AUC-ROC - ', auc_roc)


# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Нужно исследовать хотя бы 2 способа работы с дисбалансом для каждой модели. С уменьшенной выборкой есть, а с увеличенной — нет. После этого следует добавить промежуточный вывод о сравнении моделей с разными методами борьбы с дисбалансом и выбрать лучшую.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента: </b> Добавил этот этап выше, сразу после подбора гиперпараметров для техники upsampled. Вывод по исследованию результатов, добавляю ниже. </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> 👍👏
# </div>

# Сравнения результатов обучения модели на выборках с учетом использования разных методов, борьбы с дисбалансом:
# 
# Логистическая регрессия:
# Результат, после использования техники- upsampled: 
# F1: 0.5246753246753246
# AUC-ROC -  0.7938312887969258
# 
# Результат, после использования техники- downsampled: 
# F1: 0.5286343612334803
# AUC-ROC -  0.792818257956449
# 
# Дерево решений:
# Результат, после использования техники- upsampled: 
# Лучшая глубина- 6 и результат- 0.575925925925926
# AUC-ROC -  0.8234278007685487
# 
# Результат, после использования техники- downsampled: 
# Лучшая глубина- 5 и результат- 0.5900681596884129
# AUC-ROC -  0.8440425165040889
# 
# Случайный лес:
# Результат, после использования техники- upsampled: 
# Лучшая глубина- 11 , лучшее количество деревьев-  60 и лучший результат- 0.6483390607101948
# AUC-ROC -  0.8708277909153612
# 
# Результат, после использования техники- downsampled: 
# Лучшая глубина- 9 , лучшее количество деревьев-  60 и лучший результат- 0.6333973128598848
# AUC-ROC -  0.8686431544979801
# 

# Исправление дисбаланса классов, помогло значительно поднять, уровень метрика f1 и AUC-ROC в обеих моделях, при этом:
# - Гиперпараметры наиболее подходящей модели по результатам перебора, на данных обработанных после downsampled и upsampled отличались во всех случаях
# - Метрика AUC - ROC во всех случаях увеличивалась пропорционально значению метрики f1
# - Интересно также, что колличество деревьев, в наилучшей модели для данных обработанных техникой upsampled и downsampled, также совпадало, в этой связи, можно выдвинуть гипотезу, что наилучшее колличество деревьев менее зависимо от изменения размеров выборки, нежели глубина
# - Дерево решений и Случайный лес, показали лучший результат (если судить по f1 и AUC-ROC) на данных, обработанных техникой downsampled, однако наиболее высокий результат из всех моделей, показала именно модель Случайного леса, после обработки данных, техникой upsampled, из чего потенциально можно сделать вывод, что downsampled, работает лучше для моделей случайного дерева и логистической регрессии за счет уменьшения выборки, в то время как возможно за счет большего колличества гипарпараметров случайный лес, показывает лучшие результаты в случае увеличения колличества данных, техникой downsampled
# - Будет интересно проверить, обе наилучшие модели по версии техник downsampled и upsampled, чтобы понять не был ли результат на валидационной выборке случайным и действительно ли выбранная модель, лучшая.

# P.S. Вопрос в Ревьюэру, правильно ли называть эти техники борьюы с дисбалансом, downsampled и upsampled, для удобства записи или нет? Или же это только промежуточный этап и далее мы будем использовать более сложные техники борьюы вместо этих двух, у которых будут другие названия? + Вопрос что особенное я должен был выявить в изменение значения метрики AUC ROC, не до конца пока понимаю, зачем было следить за изменениями в ее значении

# <div class="alert alert-warning">
# <b>Комментарий ревьюера v2⚠️:</b> Да, так и называют, или по другому — метод увеличения и уменьшения выборки. В библиотеке  imblearn есть функции <a href="https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html">SMOTE</a> и <a href="https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html">RandomUnderSampler</a>. Вместо функций из тренажёра, в будущем можно использовать их.
#     
# Функции из тренажёра подходят только для случаев, когда перебор гиперпараметров делается циклами. А если применяется кросс-валидация, то для борьбы с дисбалансом следует использовать функции выше. Тема кросс-валидации будет рассматриваться в следующем спринте, там подробней будет рассказано про её особенности.
#     
# Идея использования AUC ROC во всех моделях заключается в том, чтобы показать, что оценка модели только одной метрикой может приводить к ошибочным результатам, т.к. нужно учитывать наличие дисбаланса. В логрегрессии это хорошо видно: в самой первой модели auc roc показывает неплохой результат, но f1 при этом очень низкая, если бы мы ориентировались только на auc roc, мы бы могли ошибочно рекомендовать эту модель как самую простую и эффективную. Далее видим, что при включении методов балансировки, f1 растёт (но недостаточно высокая), а auc roc остаётся примерно на одном уровне. Это намекает нам на то, что такой тип модели в принципе плохо находит закономерности и предсказывает эти данные. С деревом и лесом наблюдается другая картина: auc roc вместе с f1 в зависимости от метода балансировки или увеличивается или уменьшается. Увеличение качества по обеим оценкам говорит нам о хорошем обучении модели и прогнозах.
# </div>

# ## Тестирование модели

# best_model = None
# best_result = 0
# best_est = 0
# best_depth = 0
# 
# for est in range(10, 80, 10):
#     for depth in range (1, 30):
#         model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth) 
#         model.fit(features_downsampled, target_downsampled) 
#         predictions_valid = model.predict(features_test)
#         result = f1_score(target_test, predictions_valid)
#         if result > best_result:
#             best_model = model
#             best_result = result
#             best_est = est
#             best_depth = depth
# 
#         
# probabilities_valid = best_model.predict_proba(features_test)
# probabilities_one_valid = probabilities_valid[:, 1]
# 
# auc_roc = roc_auc_score(target_test, probabilities_one_valid)  
#             
# print('Лучшая глубина-', best_depth, ', лучшее количество деревьев- ', best_est, 'и лучший результат-', best_result)  
# print('AUC-ROC - ', auc_roc)

# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> На этапе тестирования не нужно делать подбор гиперпараметров, лучшие гиперпараметры уже определены в прошлом разделе, нужно использовать их. Ячейку сверху нужно убрать и использовать ту, что ниже.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента: </b> Благодарю, закоментировал данные выше. </div>

# model = RandomForestClassifier(random_state=12345, n_estimators= 40, max_depth= 17) 
# model.fit(features_downsampled, target_downsampled) 
# predictions_test = model.predict(features_test)
# result = f1_score(target_test, predictions_valid)
# 
# probabilities_valid = best_model.predict_proba(features_test)
# probabilities_one_valid = probabilities_valid[:, 1]
# 
# auc_roc = roc_auc_score(target_test, probabilities_one_valid) 
# 
# print('Результат метрики F1-', result)  
# print('AUC-ROC - ', auc_roc)

# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> При расчёте <code>result</code> вторым аргументам нужно поставить переменную <code>predictions_test</code>. И вместо <code>best_model</code> лучше поставить <code>model</code>, которую ты рассчитываешь в этой же ячейке, так ты избежишь ошибок в случае другой последовательности ячеек, где у тебя расчёт best_model.
#     
# Также не забудь добавить финальный вывод по проекту.
# </div>

# <div class="alert alert-info"> <b>Комментарий студента: </b> Благодарю, исправил ниже, не обратил внимание на это в прошлый раз. </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера v2✔️:</b> Прекрасно! Теперь всё верно 👍👏
# </div>

# Протестируем результаты работы лучшей модели на тестовой выборке

# In[24]:


model = RandomForestClassifier(random_state=12345, n_estimators= 60, max_depth= 11) 
model.fit(features_upsampled, target_upsampled) 
predictions_test = model.predict(features_test)
result = f1_score(target_test, predictions_test)

probabilities_valid = model.predict_proba(features_test)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_test, probabilities_one_valid) 

print('Результат метрики F1-', result)  
print('AUC-ROC - ', auc_roc)


# А также, в ркамках экспреримента, результаты работы модели, показавшей наибольший результат на данных обработанных с помощью техники- downsampled

# In[25]:


model = RandomForestClassifier(random_state=12345, n_estimators= 60, max_depth= 9) 
model.fit(features_downsampled, target_downsampled) 
predictions_test = model.predict(features_test)
result = f1_score(target_test, predictions_test)

probabilities_valid = model.predict_proba(features_test)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_test, probabilities_one_valid) 

print('Результат метрики F1-', result)  
print('AUC-ROC - ', auc_roc)


# ФИНАЛЬНЫЕ ВЫВОДЫ:
#     - Лучшей моделью для предсказания результата на данном датасете оказалось, модель случайного леса, обученная на данных, обработанных техникой- upsampled, со занчением макс. глубины- 11 и кол-вом деревьев- 60
#     - Значение метрики AUC-ROC, всегда росло пропорционально значению метрики F1
#     - Проверка датасета на пропуски, является крайне важным фактором на стадии предобработки данных, как и удаление лишних и неинформативных столбцов из выборки, на момент начала обучения модели, по итогу это сильно помогает повысить качество ее обучения
#     - Преобразование категориальных переменных лучше проводить после разделения выборки
#     

# ВОПРОС РЕВЬЭРУ
# Чтобы я ни делал, на тестовой выборке я получаю крайне низкое значение, хотя валидационные выборки на этой модели, давали хороший результат
# Правильно ли я понимаю, что виной всему здесь дисбаланс классов и чтобы получить нужную метрику, мне нужно исправить дисбаланс также и в тестовой выборке или этого не надо делать и смысл работы заключается именно в том, чтобы моя модель показывала хороший результата и на несбалансированной выборке?

# <div class="alert alert-warning">
# <b>Комментарий ревьюера ⚠️:</b> Дисбаланс убирается только на обучающей выборке, чтобы модель могла надёжно предсказывать оба класса. Виной низкого качества на тесте является то, что ты не сделал стандартизацию тестовой выборки, об этом был комментарий выше.
# </div>

# In[26]:


target_zeros = target_test[target == 0]
target_ones = target_test[target == 1]
print('Нулей в тестовой выборке- ', target_zeros.shape)
print('Единиц в тестовой выборке- ', target_ones.shape)


# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнен шаг 1: данные подготовлены
# - [x]  Выполнен шаг 2: задача исследована
#     - [x]  Исследован баланс классов
#     - [x]  Изучены модели без учёта дисбаланса
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 3: учтён дисбаланс
#     - [x]  Применено несколько способов борьбы с дисбалансом
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 4: проведено тестирование
# - [x]  Удалось достичь *F1*-меры не менее 0.59
# - [x]  Исследована метрика *AUC-ROC*

# <div class="alert alert-danger">
# <b>Итоговый комментарий ревьюера ❌:</b> Буду ждать следующих исправлений:
# <ul>
#     <li>удаление неинформативных признаков;</li>
#     <li>кодировка категориальных признаков;</li>
#     <li>стандартизация тестовой выборки;</li>
#     <li>вывод по исследованию баланса классов;</li>
#     <li>параметр в методе downsample;</li>
#     <li>добавление второго метода борьбы с дисбалансом;</li>
#     <li>промежуточный вывод в разделе борьбы с дисбалансом;</li>
#     <li>тестирование лучшей модели;</li>
#     <li>добавление финального вывода.</li>
# </ul>
#     
# Если возникнут вопросы, обязательно пиши!
# </div>

# <div class="alert alert-success">
# <b>Итоговый комментарий ревьюера v2✔️:</b> Ты отлично постарался! Главная цель — довести метрику F1 до 0.59 на тестовой выборке — достигнута👍 Все красные замечания исправлены и  прислушался к небольшим замечаниям тоже.
# 
# Работа принимается. Желаю тебе успехов в освоении профессии Data Science и удачи в будущих проектах!😉
# </div>
