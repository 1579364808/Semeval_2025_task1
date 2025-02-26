import openai
import pandas as pd
import time
import ast

# 设置OpenAI API密钥和自定义域名
openai.base_url = "Your domain Here"
openai.api_key = "Your API Key Here"

# 定义生成相同类型句子的函数
def generate_same_type_sentence(sentence, compound, sentence_type):
    if sentence_type == "idiomatic":
        prompt = f"Generate a new sentence that includes '{compound}' and is used idiomatically, similar to: {sentence}. Provide only the new sentence without any additional text or explanation."
    elif sentence_type == "literal":
        prompt = f"Generate a new sentence that includes '{compound}' and is used literally, similar to: {sentence}. Provide only the new sentence without any additional text or explanation."
    else:
        raise ValueError("Invalid sentence_type. Must be 'idiomatic' or 'literal'.")

    response = openai.chat.completions.create(
        model="deepseek-v3",  # 使用DeepSeek模型
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.6
    )
    same_type_sentence = response.choices[0].message.content.strip()
    return same_type_sentence

# 定义生成相反类型句子的函数
def generate_opposite_type_sentence(sentence, compound, sentence_type):
    if sentence_type == "idiomatic":
        prompt = f"Generate a new sentence that includes '{compound}' but is used literally, opposite to: {sentence}. Provide only the new sentence without any additional text or explanation."
        opposite_type = "literal"
    elif sentence_type == "literal":
        prompt = f"生成一个包含 '{compound}' 但以惯用方式使用的新句子，与以下句子相反：{sentence}。仅提供新句子，不要添加任何额外文本或解释。"
        opposite_type = "idiomatic"
    else:
        raise ValueError("Invalid sentence_type. Must be 'idiomatic' or 'literal'.")

    response = openai.chat.completions.create(
        model="deepseek-v3",  # 使用DeepSeek模型
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.6
    )
    opposite_type_sentence = response.choices[0].message.content.strip()
    return opposite_type_sentence, opposite_type

# 定义带重试和延时的生成函数
def generate_sentence_with_retry(sentence, compound, sentence_type, max_retries=10):
    for attempt in range(max_retries):
        try:
            same_type_sentence = generate_same_type_sentence(sentence, compound, sentence_type)
            opposite_type_sentence1, opposite_type1 = generate_opposite_type_sentence(sentence, compound, sentence_type)
            opposite_type_sentence2, opposite_type2 = generate_opposite_type_sentence(sentence, compound, sentence_type)
            return same_type_sentence, opposite_type_sentence1, opposite_type_sentence2, opposite_type1, opposite_type2
        except openai.RateLimitError:
            print(f"Rate limit reached. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise Exception("Max retries reached. Please try again later.")

# 定义重新排序 expected_order 的函数
def reorder_expected_order(expected_order):
    order_list = ast.literal_eval(expected_order)  # 将字符串解析为列表
    # 按照 3，2，1，0，4 的顺序重新排列
    reordered = [order_list[3], order_list[2], order_list[1], order_list[0], order_list[4]]
    return str(reordered)

# 处理单个文件
def process_file(input_file, output_file):
    # 读取TSV文件（指定分隔符为制表符，第一行为header）
    df = pd.read_csv(input_file, sep='\t', header=0)

    # 对每一行生成相同类型和相反类型的句子并增加数据数量
    new_rows = []
    for index, row in df.iterrows():
        print(f"处理第{index+1}行")
        # 生成相同类型和相反类型的句子
        same_type_sentence, opposite_type_sentence1, opposite_type_sentence2, opposite_type1, opposite_type2 = generate_sentence_with_retry(row['sentence'], row['compound'], row['sentence_type'])

        # 创建新的行并添加到列表中
        new_row_same_type = row.copy()
        new_row_same_type['sentence'] = same_type_sentence
        new_row_same_type['sentence_type'] = row['sentence_type']  # 保持句子类型不变
        new_rows.append(new_row_same_type)

        new_row_opposite_type1 = row.copy()
        new_row_opposite_type1['sentence'] = opposite_type_sentence1
        new_row_opposite_type1['sentence_type'] = opposite_type1  # 更新句子类型
        # 只有在生成相反类型的句子时才重新排序 expected_order
        new_row_opposite_type1['expected_order'] = reorder_expected_order(new_row_opposite_type1['expected_order'])
        new_rows.append(new_row_opposite_type1)

        new_row_opposite_type2 = row.copy()
        new_row_opposite_type2['sentence'] = opposite_type_sentence2
        new_row_opposite_type2['sentence_type'] = opposite_type2  # 更新句子类型
        # 只有在生成相反类型的句子时才重新排序 expected_order
        new_row_opposite_type2['expected_order'] = reorder_expected_order(new_row_opposite_type2['expected_order'])
        new_rows.append(new_row_opposite_type2)

    # 创建新的DataFrame
    new_df = pd.DataFrame(new_rows)

    # 合并原始数据和生成的数据
    merged_df = pd.concat([df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates()

    # 保存到新的TSV文件（保留header）
    merged_df.to_csv(output_file, sep='\t', index=False, header=True)
    print(f"数据生成完成并保存到 {output_file} 文件中。")

# 处理 subtask_a_train_PT.tsv 文件
process_file('subtask_a_train_PT.tsv', 'subtask_a_train_PT_ag.tsv')

# 处理 subtask_a_train.tsv 文件
process_file('subtask_a_train.tsv', 'subtask_a_train_ag.tsv')