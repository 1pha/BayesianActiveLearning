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
        except:
            logger.warn(f"Failed to load Spacy Tokenizer, {self.config.spacy_model}")
            raise

    def tokenize(self, sentence) -> torch.Tensor:

        if not hasattr(self, "tokenizer"):
            self.setup_spacy()

        if isinstance(sentence, str):

            # single instance will return tensor
            tokenized = self.tokenizer(sentence)
            return tokenized._.trf_data.tokens["input_ids"]

    def batch_tokenize(self, sentences: list) -> list:

        return [self.tokenize(s).squeeze().tolist() for s in sentences]

    def truncate(self, input_ids):

        seq_len = len(input_ids)
        if seq_len > self.config.max_seq_len:
            input_ids = input_ids[: self.config.max_seq_len]
            seq_len = self.config.max_seq_len

        else:
            zero_seq = torch.zeros(self.config.max_seq_len, dtype=torch.long)
            zero_seq[:seq_len] = input_ids
            input_ids, zero_seq = zero_seq, input_ids

        return input_ids


def build_preprocessor(config):

    return Preprocessor(config)


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()
    preprocessor = Preprocessor(data_args)

    document = "Spacy is fucking hard to use arghggh!"
    print(preprocessor.tokenize(document))
