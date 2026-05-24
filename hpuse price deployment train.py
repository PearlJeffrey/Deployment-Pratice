import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#Sample DAta
df=pd.DataFrame({
    "size":[1000,1200,1500,1800,2200],
    "price":[50,60,75,90,100]
})

x=df[["size"]]
y=df["price"]

model=LinearRegression()
model.fit(x,y)

joblib.dump(model, "trained_housing_model.pkl")

print("Model saved succesfuly")