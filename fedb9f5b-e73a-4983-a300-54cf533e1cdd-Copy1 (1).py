#!/usr/bin/env python
# coding: utf-8

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

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

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


# Изучаем полученные данные

# In[3]:


df = pd.read_csv('/datasets/Churn.csv')
df.info()
df.head()


# Подготавливаем данные для последующего анализа, преобразуем категориальные признаки в колличественные методом OHE, а также заполняем пропуски и бесконечности из таблицы, во избежангие появления ошибки- ValueError: Input contains NaN, infinity or a value too large for dtype('float64'). в дальнейшем, при тренировки модели (происходила из-за пропусков в столбце Tenure)
# 
# Также, чтобы не плодить огромное колличество столбцов и не увеличивать время работы модели, после преобразования колличественных переменных в категориальные, удаляем столбец- Surname

# In[4]:


df = df.drop(['Surname', 'CustomerId', 'RowNumber'], axis=1)
df = df.replace((np.inf, np.nan), 0).reset_index(drop=True)


# Выделяем целевой признак, а также делим выборки на валидационную, тренировочную и тестовую для последующей работы + печатаем размерность получившихся выборок для наглядности и это может помочь вовремя заметить ошибки при разделении.
# 

# In[5]:


target = df['Exited']
features = df.яdrop('Exited', axis=1)

features_train, features_test1, target_train, target_test1 = train_test_split(
    features, target, test_size=0.4, random_state=12345,  stratify= target)
features_valid, features_test,target_valid, target_test = train_test_split(
    features_test1, target_test1, test_size=0.5, random_state=12345, stratify= target_test1)

print(f"Количество строк в target_train по классам: {np.bincount(target_train)}")
print(f"Количество строк в target_valid по классам: {np.bincount(target_valid)}")
print(f"Количество строк в target_test по классам: {np.bincount(target_test)}")


# Провожу раздельное преобразование каждой выборки с помощью OHE, отдельно для каждой выборки, после их разделения.

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
features_test[numeric] = scaler.transform(features_test[numeric])


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
# - В самом начале, в классах, был выявлен сильный дисбаланс с преобладанием значений - 0, над 1 в целевом признаке, почти в 4 раза
# - При работе с несбалансированными классами, метрику F1 не удалось поднять более 0.5 на всх проверенных моделях, несмотря на перебор гиперпараметров
# - Наихудший результат в исследовании дала модель Логистической регрессии с результатом f1, равному 0.3106457242582897, а наилучший дерево решений, при указании параметров глубины в 9 и результата метрики f1 в 0.5843971631205673
# - AUC-ROC растет вместе с F1 и показывает, наилбольшее значение на модели случайного леса
# - Интересно, что случайный лес, дал почти одинаковый результат на несбаланированной выборке, в сравнение с деревом решений, однако в то же время, метрика AUC-ROC случайного леса, получилась наибольшей, что потенциально делает ее более эффективной моделью

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


# In[20]:


target_zeros = target_downsampled[target == 0]
target_ones = target_downsampled[target == 1]
print('Нулей в новой выборке- ', target_zeros.shape)
print('Единиц в новой выборке- ', target_ones.shape)


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

# ## Тестирование модели

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
