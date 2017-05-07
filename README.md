# Kaggle Titanic

[Kaggle Titanic Problem](https://www.kaggle.com/c/titanic)

# Load & Display

- deepcopy备份原始数据
- train.info()查看数据信息，缺漏情况

# Feature Engineering

## General Process

- 查看数据分布情况（train & test）
- 看feature和预测结果（离散）的相关性
    - 离散
        - sns.pointplot(data=train, x='Embarked', y='Survived')
    - 连续
        - sns.boxplot(data=train, x='Survived', y='Age')
        - train[['feature', 'result']].corr()，也可以用heatmap打印出来，最后有例子
- 填补数据：如果有空缺

## Points

- train = train.join(pd.get_dummies(train.Embarked, prefix='Embarked'))
- stripplot和boxplot结合查看连续变量分布
- 自定义函数（或lambda函数），然后被map使用。Eg: train['Title'] = train.Name.apply(get_title)
- pd.concat([train.Title, test.Title]).unique()
- **Age和其它列求相关性，然后依据最大相关的若干列做数据划分，从而精细化补全Age数据，提升了预测结果accuracy**
- train.Age.isnull().any(), test.Age.isnull().any()查看是否数据已经都填充好了，没有遗漏
- ***sns.heatmap(train.corr(), annot=True, cmap=sns.diverging_palette(240, 10, as_cmap = True), ax=ax)***

## New Features

- Title: 从Name构造
- Age_Bin
- Age^2 没有提升
- Age_Bin^2 没有提升
- Cabin_<First_Letter>: 没有提升，反而有下降。其实数据缺陷很大，应该舍弃
- Sex_Age: 把小于16岁的人，性别定义为Child，accuracy有提升
- Fare_Bin: 没有提升
- Title_Sex: 组合2个Feature为一个新Feature，效果不明显

# Cleaning

让train和test的列信息保持一致

# Prediction

- **ROC曲线**
- **Learning Curve**
- 通过VotingClassifier做**Ensembling**

用到的模型有

- LogisticRegression
- **GradientBoostingClassifier**
- AdaBoostClassifier
- **XGBClassifier**

参考的模型还有

```
estimator_scores = {
    'LogisticRegression': cross_val_score(LogisticRegression(), X=X_all, y=y_all, cv=cv).mean(),
    'Perceptron': cross_val_score(Perceptron(), X=X_all, y=y_all, cv=cv).mean(),
    'SGDClassifier': cross_val_score(SGDClassifier(), X=X_all, y=y_all, cv=cv).mean(),

    'SVC(gamma=0.001)': cross_val_score(SVC(gamma=0.001), X=X_all, y=y_all, cv=cv).mean(),
    'LinearSVC': cross_val_score(LinearSVC(), X=X_all, y=y_all, cv=cv).mean(),

    'RandomForestClassifier(100)': cross_val_score(RandomForestClassifier(n_estimators=100), X=X_all, y=y_all, cv=cv).mean(),
    'GradientBoostingClassifier': cross_val_score(GradientBoostingClassifier(), X=X_all, y=y_all, cv=cv).mean(),
    'AdaBoostClassifier': cross_val_score(AdaBoostClassifier(), X=X_all, y=y_all, cv=cv).mean(),

    'LinearDiscriminantAnalysis': cross_val_score(LinearDiscriminantAnalysis(), X=X_all, y=y_all, cv=cv).mean(),
    'QuadraticDiscriminantAnalysis': cross_val_score(QuadraticDiscriminantAnalysis(), X=X_all, y=y_all, cv=cv).mean(),
    
    'KNeighborsClassifier(3)': cross_val_score(KNeighborsClassifier(n_neighbors = 3), X=X_all, y=y_all, cv=cv).mean(),
    'GaussianNB': cross_val_score(GaussianNB(), X=X_all, y=y_all, cv=cv).mean(),
    'DecisionTreeClassifier': cross_val_score(DecisionTreeClassifier(), X=X_all, y=y_all, cv=cv).mean(),
    
    'XGBClassifier': cross_val_score(xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05), X=X_all, y=y_all, cv=cv).mean()
}
```
