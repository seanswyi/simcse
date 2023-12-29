import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ProductNameDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        sentence_column_name: str = "productname",
        fine_ctgr_column_name: str = "FINE_CTGR_STR",
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        self.batch_size = batch_size
        self.fine_ctgr_column_name = fine_ctgr_column_name
        self.sentence_column_name = sentence_column_name
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer

        self.fine_ctgrs, self.sentences, self.input_ids = self.process()
        self.pad_token_id = self.tokenizer.pad_token_id

        self.dataloader = self.create_dataloader()

    def process(self) -> tuple[list[str], list[str], list[int]]:
        fine_ctgrs = self.data[self.fine_ctgr_column_name].tolist()
        sentences = self.data[self.sentence_column_name].tolist()
        tokenized_sentences = self.tokenizer(sentences)
        input_ids = tokenized_sentences["input_ids"]

        return fine_ctgrs, sentences, input_ids

    def collate_fn(self, batch):
        fine_ctgrs = [b[0] for b in batch]
        sentences = [b[1] for b in batch]

        input_ids = [torch.tensor(b[2]) for b in batch]
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        output = {
            "fine_ctgrs": fine_ctgrs,
            "sentences": sentences,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return output

    def create_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        fine_ctgr = self.fine_ctgrs[index]
        sentence = self.sentences[index]
        input_ids = self.input_ids[index]
        return fine_ctgr, sentence, input_ids
