import logging

import torch
import spacy

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, config):

        self.config = config
        if not hasattr(config, "spacy_model"):
            config.spacy_model = "en_core_web_trf"

        self.max_seq_len = config.max_seq_len
        self.setup_spacy()

    def setup_spacy(self):

        try:
            self.tokenizer = spacy.load(self.config.spacy_model)
            logger.info(
                f"Successfully loaded Spacy Tokenizer, {self.config.spacy_model}"
            )
            spacy.prefer_gpu()

        except:
            logger.warn(f"Failed to load Spacy Tokenizer, {self.config.spacy_model}")
            raise

    def tokenize(self, sentence) -> torch.Tensor:

        if not hasattr(self, "tokenizer"):
            self.setup_spacy()

        # single instance will return tensor
        tokenized = self.tokenizer(sentence)
        return tokenized._.trf_data.tokens["input_ids"]

    def batch_tokenize(self, sentences: list) -> list:

        return [self.truncate(self.tokenize(s).squeeze().tolist()) for s in sentences]

    def truncate(self, input_ids: list):

        seq_len = len(input_ids)
        if seq_len > self.config.max_seq_len:
            input_ids = input_ids[: self.config.max_seq_len]
            seq_len = self.config.max_seq_len

        else:
            zero_pad_len = self.config.max_seq_len - seq_len
            zero_seq = [0 for _ in range(zero_pad_len)]
            input_ids += zero_seq

        return input_ids


def build_preprocessor(config):

    return Preprocessor(config)


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()
    preprocessor = Preprocessor(data_args)

    document = "Spacy is fucking hard to use arghggh!"
    print(preprocessor.tokenize(document))
