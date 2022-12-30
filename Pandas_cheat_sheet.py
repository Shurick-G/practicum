pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scipy

import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from scipy import stats as st


# Ябучий символ из 1С  

# Операторы Pandas пишут не словами, а знаками: 
# and превращается в & (амперсанд), 
# or превращается в | (вертикальная черта), 
# а not превращается в ~ (тильда).
# Каждое отдельное сравнение необходимо окружать скобками. 
# Например, логическое выражение в Python x > 0 and x < 10 в Pandas превратится в (x > 0) & (x < 10).



# Напоследок о разнице между loc и его сокращенной записью. Вот эти две записи на примере поиска всех работающих над новой функцией:
# полная запись
print(data.loc[data['Новая функция'] == '+'])

# сокращённая запись
print(data[data['Новая функция'] == '+']) 
# Сокращённая запись вам уже встречалась, но в этом уроке мы специально про неё не говорили, чтобы вас не запутать. Её применение ограничено:
# Если мы передаем только условие (как в примере выше), то можно использовать любую из двух записей, они идентичны. Логические операторы тоже можно использовать.
# Если мы хотим указать и условие, и столбец, подойдет только полная запись с loc.
# Если мы хотим изменить значения в таблице, подойдет только полная запись с loc.



# использовать в качестве индекса "наблюдение" (первый столбец).
df = pd.read_csv('<путь к файлу>', index_col=1)  
#Тогда к строкам можно образщаьтся по имени"наблюдения":
df.loc['<Имя наблюдения>']  f.loc['<индекс строки>, <имя столбца>']

LP_registrations.loc[LP_registrations['phone'] != 0, 'Дубль'] = 'Да'

# Можно сначало логически проиндексировать строки и сохранить в переменную (список гоических операторов),
# а потом передать список в .loc[]
rows = (data['Новая функция'] == '+') & (data['Роль'] == 'разработчик')
data.loc[rows, "Роль"] = "улучшатель"


df.iloc[<номер строки>, <номер столбца>  ]
# название столбцов df в список
list(df.columns) 
# Обращение к ячейки датафрэйма
df.at['<индекс строки>', '<название столбца>'] 
df.iat[1, 2] #тоже самое, но по индексам – работае намного бычстрее, т.к. Панда параллелит итерации
#присваевание (в т.ч. и замена) названий столбцов, но только всех сразу
df.columns = ['<название элемента 1>', '<название элемента 2>', '<название элемента 3>', и т.д.]
#Обращения к элементам датафрэйма:
	#По стлбцу
df[0]
	#По срезу строк
df[1:3] # если обращаться по индексам, то третий эллемент не будет включен в срез
df['<первый эллемент>':'<третий эллемент>'] # если обращаться по именам, то третий эллемент попадет в срез
# Если имя столбца – валидный питоновский идентификатор, т.е. могло бы быть имененм переменной, то к столбцам можно обращаться через точку:
df.<имя элемента> # так можно обращаться к столбцу, но не заводить новый столбец
# фильтрация по условию:
df[df['<имя элемента>'] > 3]
# отбор значений по спискку (аналог SQLного "where x in (.......)")
df[df['user_id'].isin(<список или переменная, содержащая список>)]

#среднее по всему датафрэйму:
df.mean()
#среднее по строкам:
df.mean(axis=1)
#среднее по стобцам:
df.mean(axis='columns')
#транспанировать таблицу:
df.T
# Группировка
groups = df.groupby('<имя столбца>') #созжает объект типа groups
# обращение к группе 
groups.get_group('<имя группы>')
#получить название всех групп
groups.groups.keys()
df[<имя столбца>].value_counts() # посчитать уникальное количество значений
#пример периписи значений
df[df['UnitOfMeasure'] == 'руб/1000 куб.м.']['TariffValue'] = df[df['UnitOfMeasure'] == 'руб/1000 куб.м.']['TariffValue'] / 1000



df_reader['email1_fin'] = np.where(df_reader['email1_b'].isnull(), 
                                   df_reader['email1_a'],
                                   df_reader['email1_b'])

df['DataFrame Column'] = df['DataFrame Column'].astype(int)


res_grouped = res_grouped.set_index('Клиент')


#https://habr.com/ru/post/196980/

# Вставить столбец
df2.insert(1,'country', country) 
# В нашем случае функции передается 3 аргумент:
# номер позиции, куда будет вставлен новый столбец
# имя нового столбца
# массив значений столбца (в нашем случае, это обычный список list)

# merege, что-то вроде SQL'ского JOIN'а или MERGE'а из M (PowerQuery)
res = df2.merge(df1, 'left', on='shop')
# В качестве параметров функция принимает:
# набор данных (который будет присоединен к исходному)
# тип соединения
# поле, по которому происходит соединение
result = df_1.merge(df_2, 'left', on='<поле для джоина>') # ещё пример
# ещё пример
counterparty = registrations.merge(
                                     counterparty_by_emails[['Контрагент','СовпадениеПоПоиску', 'ДатаПервогоКонтакта']], 
                                     how='left', 
                                     left_on='email',  
                                     right_on='СовпадениеПоПоиску'
                                     
                                     )


pivot_table()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
res.pivot_table(['qty'],['country'], aggfunc='sum', fill_value = 0)
# В нашем примере функция в качестве параметров принимает:
# список столбцов, по которым будет считаться агрегированные значение
# список столбцов, которые будут строками итоговой таблицы
# функция, которая используется для агрегации
# параметр для замены пустых значений на 0


uids = uids.merge(uid_user_id, 'left', on='uid')

uids[(uids['user_id'].notnull())]

new_uids_df['created_date'] = pd.to_datetime(new_uids_df['created_date'])

# вычисление разницы дат в днях (обе даты должны быть в формате datetime)
delta = (row['next_date'] - row['Дата']).days
pd.Timedelta(1, "d")
pd.Timedelta(hours=3)
data['local_time'].round('1H')


# Замена части текста как в Excel через Ctrl + H
feed = feed.replace('https://www.sima-land.ru', '', regex=True)

# отбор значения по условию "Содержит"
to_filter = general[general['source'].str.contains('google', na = False)]

# Замена пустых значений
df = df.fillna('not set')

# Сортировка
df.sort_values(<имя столбца>)
df.sort_index()


df.describe()

df.groupby(['<столбец 1>', '<Столбец 2>']).sum()
df.groupby(['<столбец 1>', '<Столбец 2>']).agg(['sum', 'max'])

logs.groupby('source').agg({'purchase':['count', 'sum']})

# Никогда не используйте циклы в работе с большт количесвом данных
# Метод
.apply(func) 
# позволяет применить функцию ко всем элементам df, не применяя цикла (отрабатывает намного быстрее)
# Метод
.map()
# позволяет применить функцию к Series, работает аналогично apply()
#Пример:
df['Name_lenght'] = df.apply(lambda x: len(x['Name']), axis=1)


# Вариант условия если для построчного прохода
orders.loc[orders['id'] == orders['first_order_id'], 'is_this_new_order'] = 'Yes'
# Удаляем строки без контрагентов (совпвдение только по потенциалу)
counterparty = counterparty[(counterparty['Контрагент_x'] != 'пусто') | (counterparty['Контрагент_y'] != 'пусто')]
# проверка
counterparty[(counterparty['Контрагент_x'] == 'пусто') & (counterparty['Контрагент_y'] == 'пусто')]


LP_registrations.loc[
    LP_registrations['email'] == 0|LP_registrations['phone'] == 0, 
    'is_this_new_order'] = 'Yes' 

LP_registrations[(LP_registrations['email'] == 0) & (LP_registrations['phone'] == 0)]['Дубль'] = 'Да'


df.dtypes()

df['age'].notna() # обрати внимани, что df['age'].notna() – уже не  DataFrame, а Series
df[df['age']].notna() # а вот так буде обработам именно DataFrame


df['age'].isna().sum()

df.loc[df['Age'].notna(), 'Name'] # loc отфильтрует по первому значению, а вернет серию, указанную во втором


 # сортировка
 df.sort_values(['Age', 'Name'], asceding=[False, True])

 # Копирование датафрейма
 df2 = df.copy(deep=True)
 # deep=True – копируется не ссылка а все объекты целиком

 # Объединение датафреймов
 # Первый способ – конкатинация, что-то типа Append ну или Union
    cdf1 = pd.concat([df, df2])
    cdf1 = pd.concat([df, df2], axis=1) # конкотинирование по столбцам
# Способ второй – merge
-----







import seaborn as sns
import matplotlib.pyplot as plot
sns.set(rc={'figure.figsize': (16, 6)}, style='witegrid')





# Фильтры
retail[retail.UnitPrice < 0]
retail[retail.CustomerID.isna()]
retail = retail.fillna({'CustomerID' : 0})
retail[retail.CustomerID == 0]
retail = retail.astype({'CustomerID' : 'Int'})
retail = retail.query('UnitPrice > 0 ')

retail.groupby('country', as_index=False).agg({'CustomerID':'count'}) \
    .sort_values('CustomerID', ascending=False).shape


retail[retail.CustomerID > 0 ].groupby('country',as_index=False) \
    .agg({'CustomerID' : pd.Series.nunique}).sort_values('CustomerID', ascending=False)


UK_data = retail.query("country == 'United Kingdom' & CustomerID > 0")

UK_data.groupby('CustomerID', as_index=False) \
    .agg({'InvoiceNo': 'count'}) \
    .rename(columns={'InvoiceNo': 'transactions'}) \
    .sort_values('transactions', ascending=False).transactions.median()


transactions_data = UK_data.groupby('CustomerID', as_index=False) \
                            .agg({'InvoiceNo' : 'count'}) \
                            .rename(columns={'InvoiceNo': 'transactions'}) \
                            .sort_values('transactions', ascending=False).transactions

sns.distplot(transactions_data.query('transactions < 1000').transactions, kde=False)

retail.invoiceData = pd.to_datetime(retail['invoiceData'])

UK_data.resample('M').CustomerID.nunique()   # M – meens by Month
UK_data.resample('D').CustomerID.nunique()   # D – meens by Day
                                             # W –||– week
UK_data.resample('M').CustomerID.nunique().plot()


UK_data.resample('M').CustomerID.nunique().rolling(10).maen() #  rolling() – скользящее среднее, 10 – окно скольжения
UK_data.resample('M').CustomerID.nunique().rolling(10).maen().plot()




inp_list = ['John', 'Bran', 'Grammy', 'Norah'] 
 
res = ' '.join([str(item) for item in inp_list]) 
print("Converting list to atring using List Comprehension:\n")
print(res) 









# ----------------------------------- Предобработка данных -------------------------------------------
# Посмотреть колдичество пропускоы в df
df.isna().sum() 

# Заполнение пропусков нужным занчением
df['track_name '] = df['track_name'].fillna('unknown') 

# Удаление столбцов, в которых в столбцах total_cases, deaths или case_fatality_rate встречается NaN
cholera = cholera.dropna(subset=['total_cases', 'deaths', 'case_fatality_rate'], axis='columns') 

dfg[~dfg['short_url_cat'].notnull()]


# Поиск явных дубликатов
df.duplicated()
df.duplicated().sum()

# Напомним, что если вызвать метод duplicated() без подсчёта суммы, 
# то на экране будут отображены все строки. Там, где есть дубликаты, будет логическое значение True, 
# где дубликата нет — False. 
# Метод sum() воспринимает все значения True как единицы, поэтому происходит сложение всех единиц, 
# то есть находится количество дубликатов.

duplicated_df = df[df.duplicated()].head() #результат — датафрейм с дубликатами

df = df.drop_duplicates() 
df = df.drop_duplicates().reset_index()          # После удаления строчек лучше обновить индексацию: чтобы в ней не осталось пропусков
df = df.drop_duplicates().reset_index(drop=True) # Можно и не создавать столбец index. Для этого у метода reset_index() изменим специальный параметр
df = df.drop_duplicates(subset=['name'], keep='first')
df = df.drop_duplicates(subset=['name'], keep='last' )

# Способ второй "ручной поиск дубликатов"
value_counts() # анализирует столбец, выбирает каждое уникальное значение и подсчитывает частоту его встречаемости в списке
stock['item'].value_counts()

# Чтобы учесть такие дубликаты, 
# все символы в строках приводят к нижнему регистру вызовом метода lower():
str.lower()
# Жля приведения к нижнему регистру объекта Series используеться
series.str.lower()
stock['item'].str.lower()

# Поиск неявных дубликатов
# Неявные дубликаты ищут методом unique(). Он возвращает перечень уникальных значений в столбце:
tennis['name'].unique()

# Удаление неявных дубликатов
tennis['name'] = tennis['name'].replace('Roger Federer', 'Роджер Федерер')

duplicates = ['Roger Fderer', 'Roger Fdrer', 'Roger Federer'] # список неправильных имён
name = 'Роджер Федерер' # правильное имя
tennis['name'] = tennis['name'].replace(duplicates, name) # замена всех значений из duplicates на name
print(tennis) # датафрейм изменился, неявные дубликаты устранены 

# Изменение типов данных --------------------------
transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce')
# errors='coerce'– что не сможет распарсить, заменит на NaN

transactions['amount'] =transactions['amount'].astype('int')

arrivals['target_datetime'] = pd.to_datetime(arrivals['target_time'], format='%Y-%m-%dZ%H:%M:%S')
# Методом to_datetime() превратим содержимое этого столбца в понятные для Python даты.
# Для этого строку форматируют, обращаясь к специальной системе обозначений, где:
# %d — день месяца (от 01 до 31)
# %m — номер месяца (от 01 до 12)
# %Y — четырёхзначный номер года (например, 2019)
# Z — стандартный разделитель даты и времени
# %H — номер часа в 24-часовом формате
# %I — номер часа в 12-часовом формате
# %M — минуты (от 00 до 59)
# %S — секунды (от 00 до 59)
# Пример:


# Метод to_datetime() работает и с форматом unix time. 
# Первый аргумент — это столбец со временем в формате unix time, второй аргумент unit со значением 's' сообщит о том, 
# что нужно перевести время в привычный формат с точностью до секунды.
# Часто приходится исследовать статистику по месяцам: например, узнать, на сколько минут сотрудник опаздывал в среднем. 
# Чтобы осуществить такой расчёт, нужно поместить время в класс DatetimeIndex и применить к нему атрибут month:
arrivals['month'] = pd.DatetimeIndex(arrivals['date_datetime']).month

# Функция для одной строки
clients['age_group'] = clients.apply(age_group_unemployed, axis=1) 




#-----------------------------------------------------------------------------------------------------
exoplanet.groupby('discovered').count()
exo_number = exoplanet.groupby('discovered')['radius'].count()


logs['source'].value_counts() # возвращает уникальные значения и количество их упоминаний

logs[logs['email'].isna()]
print(logs[logs['email'].isna()].head())


support_log_grouped['alert_group'] = support_log_grouped['user_id'].apply(alert_group)






# Задание 11. 
# Заполните пропуски в столбце `days_employed` медианными значениями по каждого типа занятости `income_type`.
# Решение из практикума
for t in data['income_type'].unique():
    data.loc[(data['income_type'] == t) & (data['total_income'].isna()), 'total_income'] = \
    data.loc[(data['income_type'] == t), 'total_income'].median()

# Моё решение
income_type_list = list(data['income_type'].unique())

for income_type in income_type_list:
    type_median = data[data['income_type'] == income_type]['total_income'].median()
    data[data['income_type'] == income_type] = \
    data[data['income_type'] == income_type].fillna({'total_income':type_median})
# По идеи, мое решение должно работать быстрее, но список типов можно собрать прямо в цикле,
# как это сделано в практикуме. И как видно из примера, использовать list() необязательно



# Рекурсия для извлечения списка из списков
def list_from_lists(main_list, list_to_append):
    for el in main_list:
        if isinstance(el, list):
            list_from_lists(el, list_to_append)
        else:
            list_to_append.append(el)




#---------------------- Примеры из вебинара ------------------------------------------------------
df.groupby('VIP')['transported'].mean()
df.groupby('VIP')['transported'].agg(['count', 'mean'])
df.groupby('VIP')['Age'].agg(['count', 'mean'])

df['age_group'] = pd.cut(df['age_group'] 5)
df.groupby('age_group')['transported'].mean().plot()
df.groupby('VIP')['age_group'].agg(['count', 'mean'])

df.groupby('VIP')['age_group'].mean().plot()
df.groupby('VIP')['age_group'].mean().plot(ylim=0)
df.groupby('VIP')['age_group'].mean().plot(kind='bar')
df.groupby('VIP')['age_group'].mean().plot.bar()
df.groupby('VIP')['age_group'].mean().sort_values(ascending=False).plot(kind='bar')

calls.groupby(['user_id', 'month']).agg(calls=('duration', 'count'))
calls.groupby(['user_id', 'month']).agg(minutes=('duration', 'sum'))
# ------------------------------------<Название> <Столбец>, <Что сделать>
# ------------------------------------<нового столбца>

# Если всё выражение взять в скобки, его части можно преносить по строкам
# не используя обратный слэш
(
    df.groupby('VIP')['age_group'].mean()
    .sort_values(ascending=False).plot(kind='bar')
    )

df['Cabin'].str[0]
df['Cabin'].str.replace(# ...
df['Cabin'].str.split('/')
df['Cabin'].str.split('/', maxsplit=1)
# maxsplit=-1 – отсутствие ограничение на количество изменений



def get_first_part(x):
    if not pd.isna(x): # если не пропуск
        return x.split('/')[0]
    return x # если пропуск

df['Cabin'].apply(get_first_part)

#--------- Статья "10 трюков библиотеки Python Pandas, которые вам нужны" -----------------------------
# https://proglib.io/p/pandas-tricks

# map
# Это классная команда для простого преобразования данных. 
# Определяете словарь, в котором «ключами» являются старые значения, а «значениями» – новые значения:
level_map = {1: 'high', 2: 'medium', 3: 'low'}
df['c_level'] = df['c'].map(level_map)

# Выбрать строки с конкретными идентификаторами
# В SQL используем SELECT * FROM… WHERE ID в («A001», «C022»,…) 
# и получаем записи с конкретными идентификаторами. 
# Если хотите сделать то же с помощью Python библиотеки Pandas, используйте
df_filter = df['ID'].isin(['A001','C022',...])
df[df_filter]

# --------------------- Простые визуализации ------------------------------------------------------------
df['time_spent'].hist(bins=100, range = (0, 1500))


import matplotlib.pyplot as plt 

plt.ylim(-50, 500)
plt.xlim(0, 200) 

df.boxplot()
df.boxplot('total_area', 'is_it_piter')
df.boxplot('<Что изучаем>', '<В каком разрезе>')


import matplotlib.pyplot as plt
# После команды вывода графика вызывают метод show(). 
# Он позволяет посмотреть, как отличаются гистограммы с разным числом корзин: 
data.hist(bins=10)
plt.show()
data.hist(bins=100)
plt.show() 



df = pd.DataFrame({'a': [2, 3, 4, 5], 'b': [4, 9, 16, 25]})
print(df)
df.plot() 
df.plot(style='o')  # Только точки
df.plot(style='x')  # Вместо точек – крестики
df.plot(style='o-') # 'o-' - кружок и линия 
# По умолчанию, метод plot строит график используя индексы в качестве значений оси Х
df.plot(x='b', y='a', style='o-') 
df.plot(x='b', y='a', style='o-', xlim=(0, 30)) 
df.plot(x='b', y='a', style='o-', xlim=(0, 30), grid=True) 
df.plot(x='b', y='a', style='o-', xlim=(0, 30), grid=True, figsize=(10, 3)) 

# одна команда в несколько строк: не забыть заключить конструкцию в скобки 
(
    data
    .query('id == "3c1e4c52"')
    .plot(x='local_time', y='time_spent', 
          ylim=(0, 1000), style='o', grid=True, figsize=(12, 6))
)


(
    data.query('id == "3c1e4c52"')
    .pivot_table(index='date_hour', values='time_spent', aggfunc='median')
    .plot(grid=True, figsize=(12, 5))
) 

print(too_fast_stat.sort_values('too_fast', ascending=False).head()) 

hw.sort_values('height').plot(x='height', y='weight') 
hw.plot(x='height', y='weight', kind='scatter') 
hw.plot(x='height', y='weight', kind='scatter', alpha=0.03) 

station_stat_full.plot.scatter(x='count', y='time_spent',  grid=True)
station_stat_full.plot.scatter(x='count', y='time_spent',  grid=True, alpha=0.1)
# Тоже самое что и:
station_stat_full.plot(x='count', y='time_spent', kind='scatter',  grid=True)


# Когда точек много и каждая в отдельности не интересна, данные отображают особым способом. 
# График делят на ячейки; пересчитывают точки в каждой ячейке. 
# Затем ячейки заливают цветом: чем больше точек — тем цвет гуще.
hw.plot(x='height', y='weight', kind='hexbin', gridsize=20, figsize=(8, 6), sharex=False, grid=True)
# gridsize – число ячеек по горизонтальной оси, аналог bins для hist().
# При столкновении с багами приходится ставить «костыли». 
# Здесь это параметр sharex=False. 
# Если значение True, то пропадёт подпись оси Х,
# а без sharex график выйдет неказистым — это «костыльный» обход бага библиотеки pandas.

# --------------------- df.query(...) ------------------------------------------------------------

print(df.query('Is_Direct == True or Has_luggage == True')) 
df.query('color == @variable') # знаком @ обозначаеться внешняя переменная

good_ids  = too_fast_stat.query('too_fast  < 0.5')
good_data = data.query('id in @good_ids.index')

# Переименование столбцов при использовании сводных таблиц и образовании "мультииндекса"
id_name.columns = ['name', 'count']

# --------------------- df.merge(...) ------------------------------------------------------------

# В отличие от merge(), по умолчанию в join() установлен тип слияния how='left'
# Ещё методом join() можно объединять больше двух таблиц: 
# их набор передают списком вместо второго датафрейма.
df1.join(df2, on='a', rsuffix='_y')['c'] 

# --------------------- Коэффициент корреляции Пирсона ------------------------------------------------------------

df['height'].corr(df['weight'])
df['weight'].corr(df['height']) # поменяли местами рост и вес 
# 0.5196907833692264
# 0.5196907833692264 

hwa.corr()
#         height    weight       age      male
# height  1.000000  0.940822  0.683689  0.139229
# weight  0.940822  1.000000  0.678335  0.155443
# age     0.683689  0.678335  1.000000  0.005887
# male    0.139229  0.155443  0.005887  1.000000 

# --------------------- Матрица диаграмм рассеяния ------------------------------------------------------------

pd.plotting.scatter_matrix(df)
pd.plotting.scatter_matrix(df, figsize=(9, 9))

# --------------------- df.where() ------------------------------------------------------------
shopping.where(shopping != 'хамон', 'обойдусь')

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)


# Метод .groupby возвращает "пары" объектов список уникальных ключей и Series значений к каждлому ключу

for developer_name, developer_data in IT_names.groupby('name'):
    print(
        'Имя {} встречается {} раза'.format(
            developer_name, len(developer_data)
        )
    ) 




spot_probs={k:spot_counts[k]/36 for k in spot_counts} # dictionary comprehension 

# Привести названия к нажнему регистру можно командой 
df.columns.str.lower(). 

# Для удобства можно еще сразу все данные вывести в процентах по пропущенным значениям. 
data.isna().sum()/len(data)*100` или `data.isna().mean()
# Возможно это будет более оптимально и позволит сразу увидеть где максимальное/минимальное 
# количество пропусков, и в каких колонках похожий процент пропусков.

# Техническую строку `<AxesSubplot:...>` можно убрать с помощью метода 
plt.show() 
# или добавления в конце кода `;`



# --------------------- DecisionTreeClassifier() ------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(features, target) 

# Параметры умодели:
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
max_features=None, max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2, splitter='best')

answers = model.predict(new_features) 

# указываем случайное состояние (число)
model = DecisionTreeClassifier(random_state=12345)

# обучаем модель как раньше
model.fit(features, target)





import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train_data.csv')

df.loc[df['last_price'] >  5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target) 


# В библиотеке sklearn метрики находятся в модуле sklearn.metrics. 
# Вычисляется accuracy функцией accuracy_score() (англ. «оценка правильности»).
from sklearn.metrics import accuracy_score 

# Функция принимает на вход два аргумента: 
# 1) правильные ответы, 2) предсказания модели. Возвращает она значение accuracy.
accuracy = accuracy_score(target, predictions) 




# ----- Пример -------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# < импортируйте функцию расчёта accuracy из библиотеки sklearn >
from sklearn.metrics import accuracy_score 
df = pd.read_csv('/datasets/train_data.csv')
df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data.csv')

test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']

train_predictions = model.predict(features)
test_predictions = model.predict(test_features)

print("Accuracy")
print("Обучающая выборка:", accuracy_score(target,      train_predictions))
print("Тестовая выборка:",  accuracy_score(test_target, test_predictions))


# ----- Деление на две выборки -------
from sklearn.model_selection import train_test_split 
df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345) 
# Напомним: в random_state мы могли записать всё что угодно, главное не None.



# --------------------- RandomForestClassifier() ------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Чтобы управлять количеством деревьев в лесу, пропишем гиперпараметр n_estimators 
# (от англ. number of estimators, «количество оценщиков»

model = RandomForestClassifier(random_state=12345, n_estimators=3)
model.fit(features, target)
model.predict(new_item)

# Правильность модели мы проверяли функцией accuracy_score(). 
# Но можно — и методом score(). Он считает accuracy для всех алгоритмов классификации.
model.score(features, target)

# --------------------- LogisticRegression() ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
# Запишите модель в переменной, указав гиперпараметры. Для постоянства результата задайте random_state, равный 12345. 
# Добавьте дополнительные гиперпараметры: solver='lbfgs' и max_iter=1000. 
# Первый гиперпараметр позволяет выбрать алгоритм, который будет строить модель. 
# Алгоритм 'lbfgs' — один из самых распространённых. 
# Он подходит для большинства задач. Гиперпараметром max_iter задаётся максимальное количество итераций обучения. 
# Значение этого параметра по умолчанию равно 100, но в некоторых случаях понадобится больше итераций.
model = LogisticRegression(random_state=12345, solver='lbfgs', max_iter=1000)
model.fit(features, target)
model.predict(new_item)
model.score(features, target)


# --------------------- mean_squared_error()  MSE  -----------------------------------------------------------
from sklearn.metrics import mean_squared_error

answers = [623, 253, 150, 237]
predictions = [649, 253, 370, 148]

result = mean_squared_error(answers, predictions)



import pandas as pd
from sklearn.metrics import mean_squared_error
df = pd.read_csv('/datasets/train_data.csv')
features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000
# < найдите MSE >
predictions = pd.Series(target.mean(), index=target.index) 
print("MSE:", mse)

# --------------------- Дерево решений в регрессии  -----------------------------------------------------------
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) # отделите 25% данных для валидационной выборки

best_model = None
best_result = 10000
best_depth = 0
for depth in range(1, 6):
    model = DecisionTreeRegressor(random_state=12345, max_depth=depth) # инициализируйте модель DecisionTreeRegressor с параметром random_state=12345 и max_depth=depth
    model.fit(features_train, target_train) # обучите модель на тренировочной выборке
    predictions_valid = model.predict(features_valid) # получите предсказания модели на валидационной выборке
    result =  mean_squared_error(target_valid, predictions_valid)**0.5 # посчитайте значение метрики rmse на валидационной выборке
    if result < best_result:
        best_model = model
        best_result = result
        best_depth = depth

print("RMSE наилучшей модели на валидационной выборке:", best_result, "Глубина дерева:", best_depth)



df.dtypes


# ---------------------------------------------- One-Hot Encoding  -----------------------------------------------------------

# Для прямого кодирования в библиотеке pandas есть функция 
pd.get_dummies(drop_first=False)


# Обучая логистическую регрессию, вы можете столкнуться с предупреждением библиотеки sklearn. Чтобы его отключить, 
# укажите аргумент solver='liblinear' (англ. solver «алгоритм решения»; library linear, «библиотека линейных алгоритмов»): 
model = LogisticRegression(solver='liblinear') 




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance.csv')

data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)
# < напишите код здесь >
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

model =  LogisticRegression(random_state=12345, solver='liblinear', max_iter=1000)
model.fit(features_train, target_train)

print("Обучено!")

# ------------------------------  Ordinal Encoding (от англ. «кодирование по номеру категории»)  ---------------------------------------
# Чтобы выполнить кодирование, 
# в sklearn есть структура данных OrdinalEncoder (англ. «порядковый кодировщик»). 
# Она находится в модуле sklearn.preprocessing (от англ. «предобработка»). 

from sklearn.preprocessing import OrdinalEncoder 
# 1. Создаём объект этой структуры данных.
encoder = OrdinalEncoder() 
# 2. Чтобы получить список категориальных признаков, 
# вызываем метод fit() — как и в обучении модели. 
# Передаём ему данные как аргумент.
encoder.fit(data) 
# 3. Преобразуем данные функцией transform() (англ. «преобразовать»). 
# Изменённые данные будут храниться в переменной data_ordinal (англ. «порядковые данные»).
data_ordinal = encoder.transform(data) 
# Чтобы код добавил названия столбцов, оформим данные в структуру DataFrame():
data_ordinal = pd.DataFrame(encoder.transform(data), columns=data.columns)
# Если преобразование признаков требуется лишь один раз, как в нашей задаче, 
# код можно упростить вызовом функции fit_transform() (от англ. «подогнать и преобразовать»). 
# Она объединяет функции: fit() и transform(). 
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('/datasets/travel_insurance.csv')

encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data), target = data_ordinal['Claim']

features = data_ordinal.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

# < напишите код здесь >
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

print("Обучено!")




# ------------------------------  Масштабирование признаков  ---------------------------------------
# ------------------------------  стандартизация данных  ---------------------------------------
# В sklearn есть отдельная структура для стандартизации данных — StandardScaler (от англ. «преобразователь масштаба методом стандартизации»). 
# Он находится в модуле sklearn.preprocessing. 
from sklearn.preprocessing import StandardScaler
# Создадим объект этой структуры и настроим его на обучающих данных. Настройка — это вычисление среднего и дисперсии:
scaler = StandardScaler()
scaler.fit(features_train) 
# Преобразуем обучающую и валидационную выборки функцией transform(). 
# Изменённые наборы сохраним в переменных: features_train_scaled (англ. «масштабированные признаки для обучения») 
# и features_valid_scaled (англ. «масштабированные признаки для проверки»):
features_train_scaled = scaler.transform(features_train)
features_valid_scaled = scaler.transform(features_valid)

# При записи изменённых признаков в исходный датафрейм код может вызывать предупреждение SettingWithCopy. 
# Причина в особенности поведения sklearn и pandas.  Специалисты уже привыкли игнорировать такое сообщение.
# Чтобы предупреждение не появлялось, в код добавляют строчку:
pd.options.mode.chained_assignment = None




import pandas as pd
from sklearn.model_selection import train_test_split
# < напишите код здесь >
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('travel_insurance.csv')

data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

numeric = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])

print(features_train.shape)


# ------------------------------  Метрики классификации  ---------------------------------------
# ------------------------------  Accuracy для решающего дерева  -------------------------------
accuracy_score()


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

# < напишите код здесь >
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

predicted_valid = model.predict(features_valid) # получите предсказания модели
accuracy_valid = accuracy_score(predicted_valid, target_valid)
print(accuracy_valid) 

# ------------------------------  Проверка адекватности модели  ---------------------------------------
# Чтобы оценить адекватность модели, проверим, как часто в целевом признаке встречается класс «1» или «0». 
# Количество уникальных значений подсчитывается методом value_counts(). Он группирует строго одинаковые величины
import pandas as pd
data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

# < напишите код здесь >
class_frequency = data['Claim'].value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar') 




import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

# чтобы работала функция value_counts(),
# мы преобразовали результат к pd.Series 
predicted_valid = pd.Series(model.predict(features_valid))

# < напишите код здесь >
class_frequency = predicted_valid.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')


# ------------------------------  Матрица ошибок  ---------------------------------------
# Матрица неточностей находится в знакомом модуле sklearn.metrics. 
# Функция confusion_matrix() принимает на вход верные ответы и предсказания, а возвращает матрицу ошибок.



# ------------------------------  Увеличение выборки  ---------------------------------------

answers = [0, 1, 0]
print(answers)
answers_x3 = answers * 3
print(answers_x3) 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

# < сделайте функцию из кода ниже >
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    return features_upsampled, target_upsampled 
    
# < добавьте перемешивание >
features_upsampled, target_upsampled = upsample(features_train, target_train, 10)
features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)

print(features_upsampled.shape)
print(target_upsampled.shape)


# ------------------------------  Уменьшение выборки  ---------------------------------------
# Преобразование проходит в несколько этапов:
# 1. Разделить обучающую выборку на отрицательные и положительные объекты;
# 2. Случайным образом отбросить часть из отрицательных объектов;
# 3. С учётом полученных данных создать новую обучающую выборку;
#    Перемешать данные. Положительные не должны идти следом за отрицательными: алгоритмам будет сложнее обучаться.

# Чтобы выбросить из таблицы случайные элементы, примените функцию sample(). 
# На вход она принимает аргумент frac (от англ. fraction, «доля»). 
# Возвращает случайные элементы в таком количестве, чтобы их доля от исходной таблицы была равна frac.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    # < напишите код здесь >
    features_downsampled = pd.concat([features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.1)
features_downsampled, target_downsampled = shuffle(features_downsampled, target_downsampled, random_state=12345)

print(features_downsampled.shape)
print(target_downsampled.shape)

# ------------------------------ Изменение порога  ---------------------------------------
# В библиотеке sklearn вероятность классов вычисляет функция 
# predict_proba() (от англ. predict probabilities, «предсказать вероятности»). 
# На вход она получает признаки объектов, а возвращает вероятности:
probabilities = model.predict_proba(features)
# Для решающего дерева и случайного леса в sklearn тоже есть функция:
predict_proba()




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

# < напишите код здесь >
probabilities_valid  = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1] 

print(probabilities_one_valid[:5])




# ------------------------------ PR-кривая  ---------------------------------------
# ------------------------------ построение кривой --------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Кривая Precision-Recall')
plt.show()

# ------------------------------ ROC-кривая  ---------------------------------------
# Чтобы выявить, как сильно наша модель отличается от случайной, посчитаем площадь под 
# ROC-кривой — AUC-ROC (от англ. Area Under Curve ROC, «площадь под ROC-кривой»). 
# Это новая метрика качества, которая изменяется от 0 до 1. AUC-ROC случайной модели равна 0.5.
# Построить ROC-кривую поможет функция roc_curve() (англ. ROC-кривая) из модуля sklearn.metrics:
from sklearn.metrics import roc_curve
# На вход она принимает значения целевого признака и вероятности положительного класса. 
# Перебирает разные пороги и возвращает три списка: значения FPR, значения TPR и рассмотренные пороги.
fpr, tpr, thresholds = roc_curve(target, probabilities)



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# < напишите код здесь >
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])
fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid) 

plt.figure()

# < постройте график >

plt.plot(fpr, tpr)
# ROC-кривая случайной модели (выглядит как прямая)
plt.plot([0, 1], [0, 1], linestyle='--')

# < примените функции plt.xlim() и plt.ylim(), чтобы
#   установить границы осей от 0 до 1 >
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])

# < примените функции plt.xlabel() и plt.ylabel(), чтобы
#   подписать оси "False Positive Rate" и "True Positive Rate" >
plt.xlabel('False Positive Rate')
plt.ylabel('rue Positive Rate')

# < добавьте к графику заголовок "ROC-кривая" функцией plt.title() >
plt.title('ROC-кривая')

plt.show()

# ------------------------------ расчет AUC-ROC ---------------------------------------
from sklearn.metrics import roc_auc_score 



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# < напишите код здесь >
from sklearn.metrics import roc_auc_score 

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)


print(auc_roc)



# ------------------------------ Метрики регрессии ---------------------------------------
# -------------------------- Коэффициент детерминаци (R2)-------------------------------------
# Коэффициент детерминации, или метрика R2 (англ. coefficient of determination; R-squared), 
# вычисляет долю средней квадратичной ошибки модели от MSE среднего, а затем вычитает эту величину из единицы. 
# Увеличение метрики означает прирост качества модели. 
# Формула расчёта R2 выглядит так:
R2 = 1 - (MSE модели / MSE среднего)
# – Значение метрики R2 равно единице только в одном случае, если MSE нулевое. Такая модель предсказывает все ответы идеально.
# –  R2 равно нулю: модель работает так же, как и среднее.
# –  Если метрика R2 отрицательна, качество модели очень низкое.
# –  Значения R2 больше единицы быть не может.
from sklearn.metrics import r2_score
r2_score(target_valid, predicted_valid)


# __________________________________________________________________________________________________________________________
Максимизация R2: поиск модели
Время для практики.
Вы найдёте модель с наибольшим значением R2. Поэкспериментируйте в Jupyter Notebook и доведите эту метрику до 0.14.  
Алгоритм решения задачи:
Подготовьте библиотеки, данные и признаки — features и target. Разделите тестовую и обучающую выборки:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'] , axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) 
Вычислите значение R2 функцией score().  R2 — метрика по умолчанию у моделей регрессии в sklearn:
model = LinearRegression()
model.fit(features_train, target_train)
print(model.score(features_valid, target_valid)) 
0.09710497146204988 
Лучший результат в этой задаче даёт алгоритм случайного леса. Найдите подходящие гиперпараметры: глубину дерева, число деревьев.
Чем больше деревьев, тем дольше учится модель. Поэтому сначала подберите глубину леса при небольшом числе деревьев:
for depth in range(1, 16, 1):
    model = RandomForestRegressor(n_estimators=20, max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    # < напишите код здесь > 
Затем увеличивайте количество деревьев:
model = RandomForestRegressor(n_estimators=60, 
    max_depth=# < напишите код здесь >, random_state=12345)
model.fit(features_train, target_train)
print(model.score(features_train, target_train))
print(model.score(features_valid, target_valid)) 
Измерьте время обучения. Тренировка одной модели займёт секунды, а поиск оптимальных гиперпараметров в цикле — несколько минут.
В Jupyter Notebook время работы ячейки измеряет команда  %%time:
%%time

model = RandomForestRegressor(n_estimators=100, random_state=12345)
model.fit(features_train, target_train) 
CPU times: user 48 s, sys: 928 ms, total: 48.9 s
Wall time: 54.9 s 
Последняя строка Wall time (от англ. wall clock time, «время настенных часов») покажет время выполнения ячейки. 
В этом уроке нет проверки кода. Но чтобы решение прошло тесты, выполните два условия:
Не удаляйте первую ячейку тетради Jupyter.
Wall time обучения одной модели должно быть меньше 50 секунд.
Запомните алгоритм и гиперпараметры. Они пригодятся в следующем уроке, где вы обучите и проверите наилучшую модель.