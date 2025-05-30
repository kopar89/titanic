import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

def convert(X):
    new_sex = []
    for i in X:
        if i == 'male':
            new_sex.append(0)
        elif i == 'female':
            new_sex.append(1)
    X = new_sex  # Заменяем всю колонку
    return X

data = pd.read_excel('train.xlsx')
target_y = ['Survived']
train_x = ['Pclass', 'Sex', 'Age', 'Fare']
Y = data[target_y]
X = data[train_x]

#очистка пустых столбцов/строк
check_1 = X['Sex'].isnull().sum() #если 0, то все хорошо, тк 0 пустых колонок
check_2 = X['Pclass'].isnull().sum() #если 0, то все хорошо, тк 0 пустых колонок
check_3 = X['Fare'].isnull().sum() #если 0, то все хорошо, тк 0 пустых колонок
check_4 = X['Age'].isnull().sum() #если 0, то все хорошо, тк 0 пустых колонок

print(f"Пустых столбцов в колонке с полом = {check_1}")
print(f"Пустых столбцов в колонке с классом билета = {check_2}")
print(f"Пустых столбцов в колонке с тарифом = {check_3}")
print(f"Пустых стобцов в колонке с возрастом = {check_4}") # в изначальной базе данных было 177 пустых столбцов

#Исправление пустых столбцов (fillna заменяет пустоту на указанное значение)
X['Age'] = X['Age'].fillna(X['Age'].mean())
check_5 = X['Age'].isnull().sum() #если 0, то все хорошо, тк 0 пустых колонок
print(f"Пустых стобцов в колонке с возрастом = {check_5}") # в изначальной базе данных было 177 пустых столбцов

X['Sex'] = convert(X['Sex'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=45) #X, Y данные, размер, количество

#обучение модели
model = svm.LinearSVC()
model.fit(x_train, y_train.values.ravel())
pred = model.predict(x_test)

print(pred[0:10])
accuracy = model.score(x_test, y_test)
print(f"Точность модели: {accuracy:.2f}")
#print(X['Sex'])
