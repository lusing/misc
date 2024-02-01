import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

X = np.array([[20240102],[20240103],[20240104],[20240105],[20240108],[20240109]]).reshape(-1,1)
y = [2962.28,2967.25,2954.35,2929.18,2887.54,2893.25]

plt.figure()
plt.title('2024年1月上证指数走势图')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.plot(X,y,'ro')
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
date1 = np.array([[20240110]])
predicted_price = model.predict(date1)[0]
print('20240110收盘价: %.2f' % predicted_price)
