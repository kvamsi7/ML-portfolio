import pandas as pd 
import pickle 
from sklearn.model_selection import train_test_split 
import catboost
from catboost import CatBoostClassifier

print("importing data")

df = pd.read_csv('german_creditrisk_data.csv')

print("Preprocessing Data")
df.fillna("Nan",inplace = True)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state=42)
print(X_train.head())

print("Train model")
model = CatBoostClassifier(iterations = 2,
                           learning_rate = 1,
                           depth = 2)

model.fit(X_train,y_train, cat_features = ['Sex','Job','Housing','Saving accounts','Checking account','Purpose'])

print("Creating Pickle File")

pickle.dump(model,open('credit_risk_01_ml_model.pkl','wb'))

