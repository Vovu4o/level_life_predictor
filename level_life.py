import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model



def get_data(data_url):
    
    lifesat = pd.read_csv(data_url + "lifesat/lifesat.csv")
    X = lifesat[["GDP per capita (USD)"]].values
    y = lifesat[["Life satisfaction"]].values
    return X, y


if __name__ == "__main__":
    gdp_per_man = float(input("Введите размер ВВП на душу населения в выбранной стране (в долларах): "))
    url = "https://github.com/ageron/data/raw/main/"
    X, y = get_data(url)
    
    model = linear_model.LinearRegression()
    model.fit(X, y)

    X_new = [[gdp_per_man]]
    print("Уровень удовлетворённости жизни в стране: {0:.2f}".format(float(model.predict(X_new)[0][0])))
