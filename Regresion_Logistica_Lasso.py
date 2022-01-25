# Modelo de regresion logistica con Lasso
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
test_cardiaco = pd.read_csv("heart.csv")
# y predecir si tiene riesgo a problemas cardiacos
# x ( conjunto de etiquetas )
caracteristicas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                   'exang', 'oldpeak', 'slope', 'ca', 'thal']
x = test_cardiaco[caracteristicas]
y = test_cardiaco.target
las = Lasso(alpha=0.000001)
X_train, x_test, Y_train, y_test = train_test_split(
    x, y, test_size=0.29, random_state=0)
# Entrenar
las.fit(X_train, Y_train)
# Predecir
Y_pred = las.predict(x_test)
print(Y_pred)
print("EL score de prediccion es: ", r2_score(y_test, Y_pred))
plt.plot(y_test, label='True')
plt.plot(Y_pred, label='Predict')
plt.legend()
plt.show()
