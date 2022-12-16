import os
import thulac
import numpy as np

source_root = 'Classical-Modern/source'
target_root = 'Classical-Modern/target'

if not (os.path.exists('dataset/source_raw.txt') and os.path.exists('dataset/target_raw.txt')):
    for f in os.listdir(source_root):
        print("processing " + f + "...")
        source_file = os.path.join(source_root, f)
        target_file = os.path.join(target_root, f + '翻译')

        # 统计各文本中行数
        with open(source_file, "r", encoding="utf-8") as source_f:
            source_len = sum(1 for _ in source_f)
        with open(target_file, "r", encoding="utf-8") as target_f:
            target_len = sum(1 for _ in target_f)

        # 对比平行语料行数，确保一致
        assert source_len == target_len
        try:
            with open('dataset/source_raw.txt', "a+", encoding="utf-8") as source_f:
                source_f.write(open(source_file, "r", encoding="utf-8").read())
            with open('dataset/target_raw.txt', "a+", encoding="utf-8") as target_f:
                target_f.write(open(target_file, "r", encoding="utf-8").read())
        except FileNotFoundError:
            os.mkdir('dataset')

if not os.path.exists('dataset/target.txt'):
    # 对文言文本分词
    with open('dataset/source_raw.txt', 'r', encoding='utf-8') as f:
        # 目标是 白话->文言，因此将文言作为目标 target.txt
        with open('dataset/target.txt', 'w+', encoding='utf-8') as s:
            print("separating wenyan text...")
            while True:
                line = f.readline()
                if line:
                    line_seq = " ".join([char for char in line])
                    s.write(line_seq)
                else:
                    break

if not os.path.exists('dataset/source.txt'):
    # 对白话文本分词，将白话作为源语言 source.txt
    print("separating modern text...")
    sep_model = thulac.thulac(seg_only=True)
    sep_model.cut_f('dataset/target_raw.txt', 'dataset/source.txt')


def divide_train_val_dataset(path_list: list, size=10000):
    print('dividing train dataset...')
    read_path = dict(zip(['src', 'tgt'], path_list))
    root = os.path.dirname(read_path['src'])
    val_path = dict(zip(['src', 'tgt'],
                        [os.path.join(root, 'src-val.txt'),
                         os.path.join(root, 'tgt-val.txt')]))
    train_path = dict(zip(['src', 'tgt'],
                          [os.path.join(root, 'src-train.txt'),
                           os.path.join(root, 'tgt-train.txt')]))
    save_path = dict(zip(['val', 'train'], [val_path, train_path]))

    def write(dataset: str, s: str):

        with open(save_path[dataset]['src'], 'a+', encoding='utf-8') as f:
            f.write(s)
        s = tgt.readline()
        with open(save_path[dataset]['tgt'], 'a+', encoding='utf-8') as f:
            f.write(s)

    with open(read_path['src'], 'r', encoding="utf-8") as source_f:
        len = sum(1 for _ in source_f)

    idx = 0
    np.random.seed(0)
    random_idx = iter(np.sort(np.random.choice(len, size, replace=False)))
    chosen_idx = next(random_idx)

    with open(read_path['src'], 'r', encoding='utf-8') as src:
        tgt = open(read_path['tgt'], 'r', encoding='utf-8')
        for line in src:
            if idx == chosen_idx:
                # 随机选取的索引条目写入验证集
                write('val', line)
                idx += 1
                try:
                    chosen_idx = next(random_idx)
                except StopIteration:
                    # 随机选取的索引都迭代完后，选取索引置空，剩余数据写入训练集
                    chosen_idx = None
            else:
                # 未选中的条目写入训练集
                write('train', line)
                idx += 1


files = [os.path.join('dataset', ''.join([i, j])) for i in ['src', 'tgt'] for j in ['-train.txt', '-val.txt']]
files_not_exist = False in [os.path.exists(file) for file in files]

if files_not_exist:
    divide_train_val_dataset(['dataset/source.txt', 'dataset/target.txt'])

print("Preprocess Done!")
