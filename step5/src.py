import os
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="替换成api_key")


def get_result(text,shots = None):
    '''

    :param text: 要预测的文本
    :param shots: few-shot需要使用的样例集合，可能为None，可能包含两个样例，可能包含四个样例
    :return:预测结果
    '''
    prompt = "Classify the following movie review as 'positive' or 'negative',reply with only one word:'positive' or 'negative':\n"#给出要求
    messages = []
    if shots:# 如果shots不为None，则加入样例
        for example in shots:
            messages.append({"role":"user","content":prompt+f"Review: {example['text']}"})
            messages.append({"role":"assistant","content":example['label']})
    messages.append({"role":"user","content":prompt+text})
    #print(prompt)
    response = client.chat.completions.create(
        model="glm-4-flash",  # 使用glm-4-flash
        messages=
            messages,
        max_tokens= 10
    )
    return (response.choices[0].message.content)

def calculate_accuracy(test_sample,shots = None):
    num_right = 0
    num_sum = 200
    for i,sample in enumerate(test_sample):#预测前100个
        if i>= 200:
            break
        text = sample['text']
        label = sample['label']
        if shots is not None:
            predicted = get_result(text,shots=shots[i])
        else:
            predicted = get_result(text)
        print("label is {}".format(label))
        print("predict is {}\n\n".format(predicted))

        if label in predicted:
            num_right += 1
    return (num_right * 1.0)/num_sum


def curate_shot(train_sample,num_shots):
    shots = []
    shot = []
    num = 0#为200测试样例个创建few-shot
    for i,sample in enumerate(train_sample):
        if i<200:#跳过前面200个
            continue

        if num >= 200:
            break
        label = sample['label']
        shot.append(
            {
                "text":sample["text"],
                "label":label
            }
        )
        if (i-3) % num_shots == 0:#每遍历num_shots（值为2 or 4）个，就加入一个样例集合(大小为num_shots个)
            shots.append(shot)
            num += 1
            shot = []
    return shots
# 首先进行数据集的构造
#分别从neg和pos中选取500个，一共1000个，组成test_data(使用1000个的原因是，4-shot最多使用1000个)
test_data = []
pos = "aclImdb_v1.tar/aclImdb_v1/aclImdb/train/pos"
neg = "aclImdb_v1.tar/aclImdb_v1/aclImdb/train/neg"
file_name_pos = os.listdir(pos)
file_name_neg = os.listdir(neg)
for i in range(500):
    with open(f'{pos}/{file_name_pos[i]}','r',encoding='UTF-8') as f:
        test_data.append({
            "text":f.read(),
            "label":"positive"
        })
    with open(f'{neg}/{file_name_neg[i]}','r',encoding='UTF-8') as f:
        test_data.append({
            "text":f.read(),
            "label":"negative"
        })

zero_shot_accuracy = calculate_accuracy(test_data)
shots = curate_shot(test_data,2)
two_shot_accuracy = calculate_accuracy(test_data,shots)
shots = curate_shot(test_data,4)
four_shot_accuracy = calculate_accuracy(test_data,shots)

print(f"Zero-shot accuracy: {zero_shot_accuracy * 100:.2f}%")
print(f"2-shot accuracy: {two_shot_accuracy * 100:.2f}%")
print(f"4-shot accuracy: {four_shot_accuracy * 100:.2f}%")
