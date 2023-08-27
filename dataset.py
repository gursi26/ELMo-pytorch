from torch.utils.data import Dataset
import torch


class ELMoPretrainDataset(Dataset):

    def __init__(self, dataset_path: str, seq_len: int):
        text = open(dataset_path, "r").readlines()
        self.seq_len = seq_len
        self.text, self.char2idx, self.idx2char, self.word2idx, self.idx2word = self.preprocess(text)
        self.tokenized_chars, self.tokenized_words = self.tokenize(self.text)

    def tokenize(self, text: list[str]) -> list[list[int]]:
        char_tokenize = [[self.char2idx[char] for char in word] for word in text]
        word_tokenize = [self.word2idx[word] for word in text]
        return char_tokenize, word_tokenize
    
    def pad(self, sequence: list[list[int]]):
        max_word_len = len(max(sequence, key=lambda x: len(x)))
        for i in range(len(sequence)):
            pad_len = max_word_len - len(sequence[i])
            front_pad = pad_len // 2
            back_pad = pad_len - front_pad
            sequence[i] = ([0] * front_pad) + sequence[i] + ([0] * back_pad)

    def preprocess(self, text: list[str]) -> list[str]:
        to_remove = []
        for i, line in enumerate(text):
            if line == " \n":
                to_remove.append(i)
            elif "=" in line:
                to_remove.append(i)

        for idx in to_remove[::-1]:
            del text[idx]
        
        text = " ".join(text).lower()
        text = text.translate(str.maketrans('', '', '!"#$%&\'()*+-./:=?@[\\]^_`{|}~'))
        text = "".join([i for i in text if (not i.isdigit()) and i.isascii()])

        char2idx = {char: i + 1 for i, char in enumerate(sorted(list(set(list(text)))))}
        char2idx["<pad>"] = 0
        idx2char = {value: key for key, value in char2idx.items()}
        text = text.split()

        word2idx = {word: i for i, word in enumerate(sorted(list(set(text))))}
        idx2word = {value: key for key, value in word2idx.items()}
        return text, char2idx, idx2char, word2idx, idx2word
    
    def __len__(self) -> int:
        return len(self.text) - self.seq_len
    
    def __getitem__(self, idx):
        src, tgt = self.tokenized_chars[idx: idx + self.seq_len], self.tokenized_words[idx+1: idx + self.seq_len + 1]
        self.pad(src)
        return torch.tensor(src), torch.tensor(tgt)
    

class PadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, batch):
        max_len = max(batch, key=lambda x: x[0].shape[-1])[0].shape[-1]
        seq_len = batch[0][1].shape[0]
        srcs, tgts = [], []
        for src, tgt in batch:
            pad_len = max_len - src.shape[-1]
            front_pad = pad_len // 2
            back_pad = pad_len - front_pad
            srcs.append(torch.cat([torch.zeros(seq_len, front_pad), src, torch.zeros(seq_len, back_pad)], dim=-1))
            tgts.append(tgt)
        return torch.stack(srcs).type(torch.long), torch.stack(tgts)