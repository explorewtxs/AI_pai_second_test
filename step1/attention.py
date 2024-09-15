import numpy as np
np.random.seed(114514)


def scaled_dot_product_attention(Q,K,V,mask = None):
    batch_size,length,dim = Q.shape
    #1. K的转置来匹配Q的最后一个维度
    K = K.transpose(0,-1,-2)
    #2. 计算attn_score并缩放
    attn_score = np.matmul(Q,K)/np.sqrt(dim)
    '''
    for i in range(10):
        for j in range(20):
            for k in range(20):
                print(attn_score[i,j,k],end='\t')
            print("\n")
        print("\n\n")
    '''
    # 3. softmax 应用于最后一个轴计算attn_weight
    attention_weights = np.exp(attn_score) / np.sum(np.exp(attn_score), axis=-1, keepdims=True)
    #4. 应用attn_weights输出output
    output = np.matmul(attention_weights, V)
    return output, attention_weights

def multi_head_attention(embed_size, num_heads, input, mask=None):
    # 6. embed_size 确保可以等分 num_heads 份， 否则输出错误
    if embed_size % num_heads !=0 :
        print("ERROR")
    head_dim = embed_size // num_heads
    # 7. 随机初始化Wq,Wk,Wv,Wo矩阵，并对input做线性变换
    Wq = np.random.randn(embed_size, embed_size)
    Wk = np.random.randn(embed_size, embed_size)
    Wv = np.random.randn(embed_size, embed_size)
    Wo = np.random.randn(embed_size, embed_size)
    Q = np.matmul(input, Wq)
    K = np.matmul(input, Wk)
    V = np.matmul(input, Wv)
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim)
    splitQ = np.split(Q,8,axis=2)
    splitQ = [np.squeeze(m,axis=2) for m in splitQ]
    splitK = np.split(K, 8, axis=2)
    splitK = [np.squeeze(m, axis=2) for m in splitK]
    splitV = np.split(V, 8, axis=2)
    splitV = [np.squeeze(m, axis=2) for m in splitV]
    outputs = []
    attention_weights = []
    #8. 利用scaled_dot_product_attention()输出的attn_output计算
    for i in range(8):
        output,attention_weight = scaled_dot_product_attention(splitQ[i],splitK[i],splitV[i])
        outputs.append(output)
        attention_weights.append(attention_weight)
    result_output = np.stack(outputs,axis=2)
    attn_output = result_output.reshape(batch_size, seq_len, embed_size)
    final_output = np.matmul(attn_output,Wo)
    result_weights = np.stack(attention_weights,axis=1)
    #9. 返回output, attN_weights
    return final_output,result_weights


embed_size = 128
num_heads = 8
input = np.random.randn(10, 20, embed_size)
output, weights = multi_head_attention(embed_size, num_heads, input)
print(output.shape, weights.shape)
print(output[0][0][:10])
print(weights[0][0][0][:10])
