import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# 数据加载
train_data = pd.read_csv("./sourcedata.csv")
test_data = pd.read_csv("./testdata.csv")
# 数据探索
print(train_data.info())
print("-" * 30)
print(train_data.describe())
print("-" * 30)
print(train_data.describe(include=["O"]))
print("-" * 30)
print(train_data.head())
print("-" * 30)
print(train_data.tail())

# 特征选择
features = ["姓名", "年龄", "花费"]
train_features = train_data[features]
train_labels = train_data["存活"]
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient="record"))
print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion="entropy")
# 决策树训练
clf.fit(train_features, train_labels)
print(test_features)
test_features = dvec.transform(test_features.to_dict(orient="record"))
print(test_features)
# 决策树预测
pred_labels = clf.predict(test_features)
print(pred_labels)
temp = pd.DataFrame(pred_labels)
temp.to_csv("./re.csv")
# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print("score 准确率为 %.4lf" % acc_decision_tree)
