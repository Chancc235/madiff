import pickle

# 读取二进制形式的pkl文件
path = "./logs/metrics.pkl"
with open(path, 'rb') as f:
    data = pickle.load(f)

# 查看内容
print(data)
