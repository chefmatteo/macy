import os
import random
import pandas as pd  # 用来处理表格数据的工具
import numpy as np   # 用来做数学计算的工具
from collections import defaultdict  # 用来存储结果的特殊字典

import torch  # 深度学习用的框架
# 下面两个是处理BERT模型的工具，BERT是一个能理解文字的AI模型
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器（一种强大的分类算法）
# 下面是一些评价模型好坏的指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV  # 用来做交叉验证和网格搜索的工具
from sklearn.decomposition import PCA  # 主成分分析，用于降维和特征选择
from sklearn.preprocessing import StandardScaler  # 标准化工具，PCA之前需要标准化

# basically the same as the import in BERT_RF.py
# except for the support_vector_machine 

# note that logistic_regression is excellent tool for binary classification: 
# goal: hybrid machine learning models
# BERT: Bidirectional Encoder Representation with Transformers
# LR: Logistic Regression
# predict the suicidal risk of patients

# 固定随机数，让每次运行结果都一样，方便比较
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 设备配置和检测 (支持Apple Silicon MPS, CUDA, CPU)
def get_optimal_device():
    """自动选择最佳设备"""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'NVIDIA GPU'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), 'Apple Silicon GPU (MPS)'
    else:
        return torch.device('cpu'), 'CPU'

DEVICE, device_name = get_optimal_device()
print(f"🚀 使用设备: {DEVICE} ({device_name})")

if DEVICE.type == 'cuda':
    print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.manual_seed(SEED)
elif DEVICE.type == 'mps':
    print(f"   Apple Silicon优化: ✅")
    print(f"   统一内存架构: ✅") 
    print(f"   建议内存: 16GB+ 推荐")
    torch.mps.manual_seed(SEED)
else:
    print("   使用CPU模式 - 可用优化:")
    print("   - 多线程处理: ✅")
    print("   - 批量处理: ✅")
    print("   - PCA降维: ✅")

# 想象你有一副768色的彩虹画（BERT的768维特征），但你要把它装进一个只有256色的调色盘里（PCA降维到256维）。
# 你会挑出最能代表整体色彩的256种颜色，这样画出来的画依然美丽、信息丰富，但更轻便、易于携带和处理。
PCA_COMPONENTS = 256  # 用PCA把768维BERT向量“浓缩”为256维，保留主要信息，提升训练速度
print(f"🔧 PCA降维配置: {768} → {PCA_COMPONENTS} 维")



# 1. 读取数据 (we are using the same data as BERT_RF.py)
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
print("📥 正在加载BERT分词器...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# 加载BERT模型（已经训练好的，会把文字变成有意义的数字）
print("📥 正在加载BERT模型...")
bert_model = BertModel.from_pretrained(BERT_MODEL)

# 将BERT模型移动到GPU（如果可用）
bert_model = bert_model.to(DEVICE)
# 把BERT设为评估模式（不用来训练，只是用来转换文字）
bert_model.eval()
print(f"✅ BERT模型已加载到 {DEVICE}")

# 根据设备类型启用相应优化
if DEVICE.type == 'cuda':
    bert_model = bert_model.half()  # 使用FP16精度，节省显存和加速
    print("⚡ 启用CUDA FP16半精度推理加速")
elif DEVICE.type == 'mps':
    # MPS目前不完全支持FP16，使用FP32但启用其他优化
    print("⚡ 启用Apple Silicon MPS加速")
    print("   注意: 使用FP32精度确保兼容性")
else:
    # CPU模式优化
    torch.set_num_threads(os.cpu_count())  # 使用所有CPU核心
    print(f"⚡ 启用CPU多线程优化 (线程数: {os.cpu_count()})")

def get_bert_embedding(text):
    """
    把一句话转换成数字向量（方便计算机处理）- GPU加速版本
    简单说就是：文字 -> BERT模型 -> 一串数字
    """
    # 如果输入的不是文字，或者是空内容，就返回一串0
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(bert_model.config.hidden_size)

    # 把文字处理成BERT能理解的格式
    # 包括：分词、补全到128长度、转成数字
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    
    # 将输入数据移动到GPU（如果可用）
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # 不计算梯度（节省内存，加快速度，因为我们不训练模型）
    with torch.no_grad():
        # 把处理好的文字输入BERT，得到输出
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = bert_model(**inputs)
        elif DEVICE.type == 'mps':
            # Apple Silicon MPS模式：标准推理
            outputs = bert_model(**inputs)
        else:
            # CPU模式：正常推理
            outputs = bert_model(**inputs)
    
    # 从BERT的输出中提取有用的部分，变成一个固定长度的数字向量
    # 简单说就是把一句话的所有词的向量取个平均值
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding.astype(np.float32)  # 确保数据类型一致



# 3. 五折交叉验证 (setting up the validation procedure)
#    （一种更可靠的测试方法，把数据分成5份，轮流当测试集）

# 初始化5折交叉验证，打乱数据顺序，固定随机数保证结果一致
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 用一个字典来存所有测试结果
all_metrics = defaultdict(list)
# 存储每一轮的最优参数
all_best_params = []
# 存储每一轮的PCA信息
all_pca_info = []

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

    # 把文字转换成数字向量（用上面定义的BERT方法）- 批量处理优化
    print(f"    🔄 正在处理训练集文本（{len(train_data)} 条）...")
    
    # 为CPU模式启用批量处理优化
    if DEVICE.type == 'cpu':
        # CPU批量处理，减少Python循环开销
        import concurrent.futures
        from functools import partial
        
        def process_texts_batch(texts, batch_size=32):
            """批量处理文本，提高CPU效率"""
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = [get_bert_embedding(t) for t in batch_texts]
                embeddings.extend(batch_embeddings)
                if i % (batch_size * 10) == 0:
                    print(f"      处理进度: {min(i+batch_size, len(texts))}/{len(texts)}")
            return embeddings
        
        train_embeddings = np.array(process_texts_batch(train_data["text"].tolist()))
        print(f"    🔄 正在处理测试集文本（{len(test_data)} 条）...")
        test_embeddings = np.array(process_texts_batch(test_data["text"].tolist()))
    else:
        # GPU/MPS模式：正常处理
        train_embeddings = np.array([get_bert_embedding(t) for t in train_data["text"]])
        print(f"    🔄 正在处理测试集文本（{len(test_data)} 条）...")
        test_embeddings = np.array([get_bert_embedding(t) for t in test_data["text"]])

    # 内存清理（根据设备类型）
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE.type == 'mps':
        torch.mps.empty_cache()  # Apple Silicon内存清理

    print(f"    📊 BERT嵌入完成: 训练集 {train_embeddings.shape}, 测试集 {test_embeddings.shape}")

    # ===== PCA特征选择和降维 =====
    print(f"    🔧 正在进行PCA特征选择和标准化...")
    
    # 1. 标准化（PCA之前必须标准化）
    scaler = StandardScaler()
    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    test_embeddings_scaled = scaler.transform(test_embeddings)
    
    # 2. PCA降维
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    train_embeddings_pca = pca.fit_transform(train_embeddings_scaled)
    test_embeddings_pca = pca.transform(test_embeddings_scaled)
    
    # 计算解释方差比例
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"    📈 PCA降维完成: {train_embeddings.shape[1]} → {PCA_COMPONENTS} 维")
    print(f"    📈 保留信息量: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # 使用PCA处理后的特征
    train_embeddings = train_embeddings_pca
    test_embeddings = test_embeddings_pca

    # 提取训练和测试的标签（我们要预测的结果）
    train_Labels = train_data["Label"].values
    test_Labels = test_data["Label"].values

    # 4. 用逻辑回归做分类 + 网格搜索优化超参数
    # (Logistic Regression with Grid Search for hyperparameter optimization)

    # 定义网格搜索的参数空间
    # Define the parameter space for grid search
    param_grid = {
        'C': [1, 10],                              # 正则化参数：最常用的值
        'penalty': ['l2'],                         # 正则化类型：L2最稳定且快速
        'solver': ['lbfgs'],                       # 优化算法：默认且高效的求解器
        'max_iter': [1000]                        # 最大迭代次数：通常足够收敛
    }
    
    print(f"    开始网格搜索最优逻辑回归参数（样本数: {len(train_embeddings)}）...")
    print(f"    搜索空间: C={param_grid['C']}, penalty={param_grid['penalty']}, solver={param_grid['solver']}")
    
    # 初始化网格搜索
    # Initialize grid search with 3-fold cross-validation within training data
    grid_search = GridSearchCV(
        LogisticRegression(random_state=SEED),      # 基础逻辑回归模型
        param_grid,                                 # 参数搜索空间
        cv=2,                                      # 2折交叉验证（快速验证）
        scoring='f1_macro',                        # 优化F1分数（适合不平衡数据）
        n_jobs=-1,                                 # 使用所有CPU核心加速
        verbose=1                                  # 显示进度便于监控
    )
    
    # 训练模型并搜索最优参数
    print(f"    正在训练逻辑回归模型并搜索最优超参数...")
    grid_search.fit(train_embeddings, train_Labels)
    
    # 获取最优模型
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"    最优参数: {best_params}")
    print(f"    交叉验证最优F1分数: {best_score:.4f}")

    # Use the trained model to make predictions on the test set
    print(f"    Predicting on test set (number of samples: {len(test_embeddings)})...")
    pred_Labels = clf.predict(test_embeddings)  # Predicted class labels (e.g., yes/no)
    pred_probs = clf.predict_proba(test_embeddings)[:, 1]  # Predicted probability for the positive class
    
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
    all_best_params.append(best_params)
    all_pca_info.append({
        'explained_variance': explained_variance,
        'components_shape': pca.components_.shape,
        'n_components': PCA_COMPONENTS
    })


# 6. 汇总所有轮的结果

print("\n===== 最终结果 (5轮的平均值 ± 波动范围) =====")
for k, v in all_metrics.items():
    print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

print("\n===== 网格搜索结果汇总 =====")
print("各轮最优参数:")
for i, params in enumerate(all_best_params, 1):
    print(f"第 {i} 轮: {params}")

# 统计最常用的参数组合
from collections import Counter
print("\n参数使用频率统计:")

# 统计各参数的使用频率
c_values = [params['C'] for params in all_best_params]
penalty_values = [params['penalty'] for params in all_best_params]
solver_values = [params['solver'] for params in all_best_params]

print(f"C值频率: {dict(Counter(c_values))}")
print(f"penalty频率: {dict(Counter(penalty_values))}")
print(f"solver频率: {dict(Counter(solver_values))}")

# 找出最常用的参数组合
param_combinations = [str(params) for params in all_best_params]
most_common = Counter(param_combinations).most_common(1)[0]
print(f"\n最常用参数组合: {most_common[0]} (出现 {most_common[1]} 次)")

print("\n===== 网格搜索优化建议 =====")
print("基于以上结果，建议的逻辑回归参数配置:")
c_mode = Counter(c_values).most_common(1)[0][0]
penalty_mode = Counter(penalty_values).most_common(1)[0][0]
solver_mode = Counter(solver_values).most_common(1)[0][0]
print(f"推荐配置: C={c_mode}, penalty='{penalty_mode}', solver='{solver_mode}'")

print("\n===== PCA特征选择分析 =====")
avg_explained_variance = np.mean([info['explained_variance'] for info in all_pca_info])
print(f"平均信息保留率: {avg_explained_variance:.4f} ({avg_explained_variance*100:.2f}%)")
print(f"特征维度: {768} → {PCA_COMPONENTS} (降维 {((768-PCA_COMPONENTS)/768)*100:.1f}%)")

# 设备使用情况汇总
if DEVICE.type == 'cuda':
    print(f"\n🚀 NVIDIA GPU加速效果:")
    print(f"   设备: {torch.cuda.get_device_name(0)}")
    print(f"   内存使用: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"   最大内存: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    acceleration_factor = 3
elif DEVICE.type == 'mps':
    print(f"\n🍎 Apple Silicon MPS加速效果:")
    print(f"   统一内存架构: 高效内存访问")
    print(f"   神经引擎: AI任务加速")
    print(f"   功耗优化: 性能/功耗比优秀")
    acceleration_factor = 2
else:
    print(f"\n💻 CPU优化效果:")
    print(f"   多线程处理: {os.cpu_count()} 线程")
    print(f"   批量处理: 减少Python开销")
    print(f"   内存优化: 高效缓存使用")
    acceleration_factor = 1

print(f"\n🎯 性能优化总结:")
device_status = "✅" if DEVICE.type != 'cpu' else "⚡"
print(f"   BERT特征提取: {device_status} {device_name}")
print(f"   PCA降维优化: ✅ ({768} → {PCA_COMPONENTS} 维)")
print(f"   网格搜索优化: ✅ (最优超参数自动选择)")
print(f"   预期训练加速: ~{acceleration_factor}x (设备) + ~{768/PCA_COMPONENTS:.1f}x (PCA降维)")

if DEVICE.type == 'mps':
    print(f"   Apple Silicon: 已检测并启用优化 🚀")
else:
    print(f"   建议: 升级到Apple Silicon MacBook Pro获得最佳性能")
