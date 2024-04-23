import re
import json


squadnqg_train_path = './data/squadnqg/train.json'
squadnqg_dev_path = './data/squadnqg/dev.json'
squadnqg_test_path = './data/squadnqg/test.json'

train_src_path = './data/raw/src_train.txt'
train_trg_path = './data/raw/trg_train.txt'

dev_src_path = './data/raw/src_dev.txt'
dev_trg_path = './data/raw/trg_dev.txt'

test_src_path = './data/raw/src_test.txt'
test_trg_path = './data/raw/trg_test.txt'


train_context=[]
train_question=[]

with open(squadnqg_train_path, 'r') as f:
    train_data = json.load(f)

for i in range(len(train_data)):
    one_train_data = train_data[i]
    two_train_data = one_train_data['paragraphs']
    for j in range(len(two_train_data)):
        context_train_data = two_train_data[j]['context']
        qas_train_data = two_train_data[j]['qas']
        for k in range(len(qas_train_data)):
            qa_train_data = qas_train_data[k]
            a_train_data = qa_train_data['answers'][0]['answer_start']
            for s1 in range(a_train_data,0,-1):
                if context_train_data[s1]==' ' and context_train_data[s1-1]=='.':
                    break
                else: 
                    continue
            for s2 in range(a_train_data, len(context_train_data), 1):
                if context_train_data[s2-1]=='.' and context_train_data[s2]==' ':
                    break
                else:
                    continue
            q_train_data = qa_train_data['question']
            if s1==1 and s2==len(context_train_data)-1 and context_train_data[s1-1:s2+1]!='':
                train_context.append(context_train_data[s1-1:s2+1])
                train_question.append(q_train_data)
            elif s1!=1 and s2==len(context_train_data)-1 and context_train_data[s1+1:s2+1]!='':
                train_context.append(context_train_data[s1+1:s2+1])
                train_question.append(q_train_data)
            elif s1==1 and s2!=len(context_train_data)-1 and context_train_data[s1-1:s2]!='':
                train_context.append(context_train_data[s1-1:s2])
                train_question.append(q_train_data)
            elif context_train_data[s1+1:s2]!='':
                train_context.append(context_train_data[s1+1:s2])
                train_question.append(q_train_data)


dev_context=[]
dev_question=[]

with open(squadnqg_dev_path, 'r') as f:
    dev_data = json.load(f)

for i in range(len(dev_data)):
    one_dev_data = dev_data[i]
    two_dev_data = one_dev_data['paragraphs']
    for j in range(len(two_dev_data)):
        context_dev_data = two_dev_data[j]['context']
        qas_dev_data = two_dev_data[j]['qas']
        for k in range(len(qas_dev_data)):
            qa_dev_data = qas_dev_data[k]
            a_dev_data = qa_dev_data['answers'][0]['answer_start']
            for s1 in range(a_dev_data,0,-1):
                if context_dev_data[s1]==' ' and context_dev_data[s1-1]=='.':
                    break
                else: 
                    continue
            for s2 in range(a_dev_data, len(context_dev_data), 1):
                if context_dev_data[s2-1]=='.' and context_dev_data[s2]==' ':
                    break
                else:
                    continue
            q_dev_data = qa_dev_data['question']
            if s1==1 and s2==len(context_dev_data)-1 and context_dev_data[s1-1:s2+1]!='':
                dev_context.append(context_dev_data[s1-1:s2+1])
                dev_question.append(q_dev_data)
            elif s1!=1 and s2==len(context_dev_data)-1 and context_dev_data[s1+1:s2+1]!='':
                dev_context.append(context_dev_data[s1+1:s2+1])
                dev_question.append(q_dev_data)
            elif s1==1 and s2!=len(context_dev_data)-1 and context_dev_data[s1-1:s2]!='':
                dev_context.append(context_dev_data[s1-1:s2])
                dev_question.append(q_dev_data)
            elif context_dev_data[s1+1:s2]!='':
                dev_context.append(context_dev_data[s1+1:s2])
                dev_question.append(q_dev_data)


test_context=[]
test_question=[]

with open(squadnqg_test_path, 'r') as f:
    test_data = json.load(f)

for i in range(len(test_data)):
    one_test_data = test_data[i]
    two_test_data = one_test_data['paragraphs']
    for j in range(len(two_test_data)):
        context_test_data = two_test_data[j]['context']
        qas_test_data = two_test_data[j]['qas']
        for k in range(len(qas_test_data)):
            qa_test_data = qas_test_data[k]
            a_test_data = qa_test_data['answers'][0]['answer_start']
            for s1 in range(a_test_data,0,-1):
                if context_test_data[s1]==' ' and context_test_data[s1-1]=='.':
                    break
                else: 
                    continue
            for s2 in range(a_test_data, len(context_test_data), 1):
                if context_test_data[s2-1]=='.' and context_test_data[s2]==' ':
                    break
                else:
                    continue
            q_test_data = qa_test_data['question']
            if s1==1 and s2==len(context_test_data)-1 and context_test_data[s1-1:s2+1]!='':
                test_context.append(context_test_data[s1-1:s2+1])
                test_question.append(q_test_data)
            elif s1!=1 and s2==len(context_test_data)-1 and context_test_data[s1+1:s2+1]!='':
                test_context.append(context_test_data[s1+1:s2+1])
                test_question.append(q_test_data)
            elif s1==1 and s2!=len(context_test_data)-1 and context_test_data[s1-1:s2]!='':
                test_context.append(context_test_data[s1-1:s2])
                test_question.append(q_test_data)
            elif context_test_data[s1+1:s2]!='':
                test_context.append(context_test_data[s1+1:s2])
                test_question.append(q_test_data)

all_context = train_context
all_context.extend(dev_context)
all_context.extend(test_context)

train_context = all_context[:len(all_context)-18000]
dev_context = all_context[len(all_context)-18000:len(all_context)-9000]
test_context = all_context[len(all_context)-9000:len(all_context)]

all_question = train_question
all_question.extend(dev_question)
all_question.extend(test_question)

train_question = all_question[:len(all_question)-18000]
dev_question = all_question[len(all_question)-18000:len(all_question)-9000]
test_question = all_question[len(all_question)-9000:len(all_question)]

def tokenizer_char(content):
    special_chars = [',', '.', '’', '\'', '“', '”', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '-', '--']
    for i in range(len(content)):
        sentence = content[i]
        for char in special_chars:
            if char == '(': 
                sentence = re.sub(rf'([{char}])', r'\1 ', sentence)
            else:
                sentence = re.sub(rf'([{char}])', r' \1', sentence)
        content[i] = sentence
    return content
 
 
train_context = tokenizer_char(train_context)
train_question = tokenizer_char(train_question)

dev_context = tokenizer_char(dev_context)
dev_question = tokenizer_char(dev_question)

test_context = tokenizer_char(test_context)
test_question = tokenizer_char(test_question)


def generate_raw(raw_path, raw_data):
    file_raw = open(raw_path, 'w', encoding=u'utf-8')
    for i in range(len(raw_data)):
        while "\n" in raw_data[i]:
            raw_data[i] = raw_data[i].replace("\n", " ")
        one_data = raw_data[i]+"\n"
        file_raw.write(one_data)
    file_raw.close()


generate_raw(train_src_path, train_context)
generate_raw(train_trg_path, train_question)

generate_raw(dev_src_path, dev_context)
generate_raw(dev_trg_path, dev_question)

generate_raw(test_src_path, test_context)
generate_raw(test_trg_path, test_question)
