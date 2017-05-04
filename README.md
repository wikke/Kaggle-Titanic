# Kaggle Titanic

[Kaggle Titanic Problem](https://www.kaggle.com/c/titanic)

## 数据可视化

- 看分布统计，比如直方图，散点图，得到一些提示
- 看和预测最终结果的相关性，比如pointplot，看这个变量是否是影响最终预测结果的因子，无关变量可以考虑剔除

## 清洗、补全数据

- 如果缺失零星数据，直接赋值为平均值，或者大多数的值即可
- 如果确实有一部分，可以按照均匀分布补齐（结合均值、标准差）
- 如果缺失很多，可以考虑放弃这个因素

## 模型预测

- LogisticRegression
- SVC
- RandomForestClassifier
- KNeighborsClassifier
- GaussianNB
