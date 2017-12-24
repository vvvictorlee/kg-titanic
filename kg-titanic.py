import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

#Preproccesing taken from https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def preprocess_data(path, test=False):
    data_train = pd.read_csv(path)
    data_train = simplify_ages(simplify_cabins(simplify_fares(format_name(data_train))))

    dummy_fields = ['Sex', 'Age', 'Cabin', 'Embarked', 'Fare', 'Pclass']
    for each in dummy_fields:
        dummies = pd.get_dummies(data_train[each], prefix=each, drop_first=False)
        data_train = pd.concat([data_train, dummies], axis=1)

    X_all = data_train.drop(['Sex', 'PassengerId', 'Ticket', 'Name', 'Lname', 'NamePrefix', 'Age', 'Cabin', 'Embarked', 'Fare', 'Pclass'], axis=1)
    if False == test:
        X_all = X_all.drop(['Survived'], axis=1)
        y_all = data_train['Survived']
        return X_all, y_all
    else:
      return X_all
    
def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0

def fix_columns( d, columns ):  
    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    #if extra_cols:
       # print("extra columns:", extra_cols)

    d = d[ columns ]
    return d

X_all, y_all = preprocess_data('data/train.csv')

# Splitting into training and validation sets
X_train = X_all.iloc[0:700]
y_train = y_all.iloc[0:700]

#print(X_train.head())

X_val = X_all.iloc[701:]
y_val = y_all.iloc[701:]

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim = 32))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
'''
model.fit(X_train.values, y_train.values,
          validation_data=(X_val.values, y_val.values), 
          epochs=40, batch_size=20,
          callbacks=[checkpointer])
'''
X_test = preprocess_data('data/test.csv', test=True)

fix_columns(X_test, X_train.columns)

#print(X_test.head())

model.load_weights('saved_models/weights.best.hdf5')
output = model.predict(X_test)

#print(output)

f = open('results.csv', 'w')
f.write('PassengerID,Survived\n')
idNum = 892
for o in output:
    f.write(str(idNum))
    f.write(',')
    if o >= 0.5:
        f.write('1\n')
    else:
        f.write('0\n')
    idNum += 1

f.close()
