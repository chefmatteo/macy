import os
import random
import pandas as pd  # 用来处理表格数据的工具
import numpy as np   # 用来做数学计算的工具
from collections import defaultdict  # 用来存储结果的特殊字典

import torch  # 深度学习用的框架
# 下面两个是处理BERT模型的工具，BERT是一个能理解文字的AI模型
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器（一种做分类的算法）
# 下面是一些评价模型好坏的指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold  # 用来做交叉验证的工具

# 固定随机数，让每次运行结果都一样，方便比较
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 1. 读取数据

# 读取文本数据文件（里面有用户ID和对应的文字内容）
text_data = pd.read_csv("BERT_RF.py/features_linguistic_cleaned.csv")
# 读取标签数据文件（里面有用户ID和对应的分类标签，比如好/坏、是/否等）
Label_data = pd.read_excel("BERT_RF.py/EMA_Dataset.xlsx")

# 把上面两个表通过"ID"合并起来，形成一个完整的数据集
data = pd.merge(text_data, Label_data, on="ID")
# 提取所有不重复的用户ID，后面会按用户来划分训练和测试数据
# 这样做是为了避免同一个人的数据既用来训练又用来测试，保证公平
users = list(set(data["ID"]))

# 打印一下数据基本情况
print(f"总用户数: {len(users)}，总样本数: {len(data)}")


# 2. 用BERT模型把文字转换成数字
#    （BERT只用来提取特征，不重新训练）

# 用的是粤语的BERT模型
BERT_MODEL = "indiejoseph/bert-base-cantonese"
# 文字最长保留128个词，太长的会截断，太短的会补全
MAX_LEN = 128

# 加载分词器（把句子拆成一个个词或字的工具）
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
# 加载BERT模型（已经训练好的，会把文字变成有意义的数字）
bert_model = BertModel.from_pretrained(BERT_MODEL)
# 把BERT设为评估模式（不用来训练，只是用来转换文字）
bert_model.eval()

def get_bert_embedding(text):
    """
    把一句话转换成数字向量（方便计算机处理）
    简单说就是：文字 -> BERT模型 -> 一串数字
    """
    # 如果输入的不是文字，或者是空内容，就返回一串0
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(bert_model.config.hidden_size)

    # 把文字处理成BERT能理解的格式
    # 包括：分词、补全到128长度、转成数字
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    
    # 不计算梯度（节省内存，加快速度，因为我们不训练模型）
    with torch.no_grad():
        # 把处理好的文字输入BERT，得到输出
        outputs = bert_model(** inputs)
    
    # 从BERT的输出中提取有用的部分，变成一个固定长度的数字向量
    # 简单说就是把一句话的所有词的向量取个平均值
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding



# 3. 五折交叉验证
#    （一种更可靠的测试方法，把数据分成5份，轮流当测试集）

# 初始化5折交叉验证，打乱数据顺序，固定随机数保证结果一致
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 用一个字典来存所有测试结果
all_metrics = defaultdict(list)

# 循环处理每一份数据（一共5份）
for fold, (train_idx, test_idx) in enumerate(kf.split(users), 1):
    # 得到这一轮用来训练和测试的用户ID
    train_users = [users[i] for i in train_idx]
    test_users = [users[i] for i in test_idx]

    # 根据用户ID，从总数据中选出训练集和测试集
    train_data = data[data["ID"].isin(train_users)]
    test_data = data[data["ID"].isin(test_users)]

    # 打印这一轮的基本信息
    print(f"\n===== 第 {fold} 轮 =====")
    print(f"训练集用户数: {len(train_users)}，测试集用户数: {len(test_users)}")

    # 把文字转换成数字向量（用上面定义的BERT方法）
    train_embeddings = np.array([get_bert_embedding(t) for t in train_data["text"]])
    test_embeddings = np.array([get_bert_embedding(t) for t in test_data["text"]])

    # 提取训练和测试的标签（我们要预测的结果）
    train_Labels = train_data["Label"].values
    test_Labels = test_data["Label"].values


    # 4. 用随机森林做分类
    #    （一种常用的分类算法，类似很多棵决策树一起投票）

    # 初始化随机森林模型，用200棵树来做决策
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    # 用训练集的数据训练模型（让模型学习文字和标签的关系）
    clf.fit(train_embeddings, train_Labels)

    # 用训练好的模型做预测
    pred_Labels = clf.predict(test_embeddings)  # 预测的类别（比如是/否）
    pred_probs = clf.predict_proba(test_embeddings)[:, 1]  # 预测是某个类别的概率
    
    # 5. 评估模型好坏

    # 计算各种评价指标
    acc = accuracy_score(test_Labels, pred_Labels)  # 准确率：预测对的比例
    # 精确率：预测为正的里面，真正为正的比例
    precision = precision_score(test_Labels, pred_Labels, average="macro", zero_division=0)
    recall = recall_score(test_Labels, pred_Labels, average="macro", zero_division=0)  # 召回率：真正为正的里面，预测对的比例
    f1 = f1_score(test_Labels, pred_Labels, average="macro", zero_division=0)  # F1值：精确率和召回率的综合指标
    auc = roc_auc_score(test_Labels, pred_probs)  # AUC值：衡量模型区分正负类的能力

    # 计算混淆矩阵的四个基本值
    # tn: 实际是负，预测也是负；fp: 实际是负，预测是正
    # fn: 实际是正，预测是负；tp: 实际是正，预测是正
    tn, fp, fn, tp = confusion_matrix(test_Labels, pred_Labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性：实际为负的里面，预测对的比例
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值：预测为负的里面，实际为负的比例

    # 打印这一轮的评价结果
    print(f"第 {fold} 轮结果 | 准确率: {acc:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, "
          f"F1值: {f1:.4f}, AUC值: {auc:.4f}, 特异性: {specificity:.4f}, 阴性预测值: {npv:.4f}")

    # 把这一轮的结果存起来，后面算平均值
    all_metrics["准确率"].append(acc)
    all_metrics["精确率"].append(precision)
    all_metrics["召回率"].append(recall)
    all_metrics["F1值"].append(f1)
    all_metrics["AUC值"].append(auc)
    all_metrics["特异性"].append(specificity)
    all_metrics["阴性预测值"].append(npv)


# 6. 汇总所有轮的结果

print("\n===== 最终结果 (5轮的平均值 ± 波动范围) =====")
for k, v in all_metrics.items():
    print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
#情况一：所有指标偏低
#考虑1、文本特征不够好 2、分类器太简单
#适当调高MAX_LEN的值，改为256或512，截取更多关键信息（改大后会计算变慢、内存占用增加）
#把n_estimators从 200 增加到 300 或 500（树越多，学习能力越强，但训练变慢）

#情况二：Accuracy还行，Precision低
#考虑是模型太“激进”，喜欢把样本预测为正类，导致很多假阳性，例如把没病的人判为生病
# 改成带阈值的预测：
#pred_probs = clf.predict_proba(test_embeddings)[:, 1]  # 正类概率
#pred_Labels = (pred_probs > 0.6).astype(int)  # 概率＞0.6才判为正类（阈值可调整）

#情况三：Accuracy还行，Recall低
#考虑模型太“保守”，很少预测正类，导致很多假阴性，例如漏判很多真生病的人
#降低分类阈值
#pred_probs = clf.predict_proba(test_embeddings)[:, 1]
#pred_Labels = (pred_probs > 0.4).astype(int)  # 阈值从0.5降到0.4

#情况四：模型在训练集表现极好，但测试集差
#考虑过拟合
#减少树的数量（n_estimators）或限制树深度（max_depth）

#情况五：指标波动大
#增加交叉验证折数，kf = KFold(n_splits=5, shuffle=True, random_state=SEED)，例如增加到10折



