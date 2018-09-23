
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 创建一个形状为(len(sequences), dimension)的零矩阵
    results =np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # 将results[i]的指定索引设为1
        results[i, sequence] = 1.
    return results

# tsa = [3, 5]
# sa = vectorize_sequences(tsa)
# print(sa)