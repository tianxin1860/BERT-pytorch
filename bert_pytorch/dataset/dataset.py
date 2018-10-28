from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np
import copy


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.seq_len = seq_len

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        #t1, (t2, is_next_label) = self.datas[item][0], self.random_sent(item)
        t1 = self.datas[item]
        t1_random, t1_label = self.random_word(t1)
        #t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        #t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        #t2 = t2_random + [self.vocab.eos_index]

        #t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        #t2_label = t2_label + [self.vocab.pad_index]

        #segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = t1_random[:self.seq_len]
        bert_label = t1_label[:self.seq_len]

        padding = [self.vocab["<PAD>"] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        #, segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        #for key, value in output.items():
        #    print ("key:{0}\tvalue:{1}".format(key, value))

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = [int(word_id) for word_id in sentence.split()]
        output_label = copy.deepcopy(tokens)

        prob = np.random.uniform(size = len(tokens))
        for i, token in enumerate(tokens):
            if prob[i] < 0.15 * 0.8:
                # 80% randomly change token to make token
                tokens[i] = self.vocab["<MASK>"]

            # 10% randomly change token to random token
            elif 0.15 * 0.8 <= prob[i] < 0.15 * 0.9:
                tokens[i] = random.randrange(len(self.vocab))

            # 10% randomly change token to current token
            elif 0.15 * 0.9 <= prob[i] < 0.15 * 1.0:
                #tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                pass
            else:
                output_label[i] = self.vocab["<PAD>"]

        return tokens, output_label

    def random_sent(self, index):
        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return self.datas[index][1], 1
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0
