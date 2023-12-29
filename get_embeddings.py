import argparse
import pickle

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from simcse.dataset import ProductNameDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="monologg/koelectra-base-discriminator",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="/data/shared/prcmd/valid.csv",
    )
    parser.add_argument(
        "--sentence_column_name",
        type=str,
        default="productname",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    return args


def get_embeddings(
    model,
    dataloader,
    device,
):
    sentences2embeddings = {}
    for batch in tqdm(
        iterable=dataloader,
        desc="Getting embeddings",
        total=len(dataloader),
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs[0][:, 0, :]

        sentence2embedding = {
            sentence: {
                "fine_ctgr": fine_ctgr,
                "embedding": embedding,
            }
            for fine_ctgr, sentence, embedding in zip(
                batch["fine_ctgrs"],
                batch["sentences"],
                cls_embeddings,
            )
        }

        sentences2embeddings.update(sentence2embedding)

    return sentences2embeddings


def main():
    args = get_args()

    save_file = os.path.splitext(args.data_file)[0]
    save_file = f"{save_file}-embeddings.pickle"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    dataset = ProductNameDataset(
        data_file=args.data_file,
        tokenizer=tokenizer,
        sentence_column_name=args.sentence_column_name,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    dataloader = dataset.dataloader

    sentences2embeddings = get_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    with open(file=save_file, mode="wb") as f:
        pickle.dump(obj=sentences2embeddings, file=f)


if __name__ == "__main__":
    main()
