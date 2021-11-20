import logging

import torch
import spacy

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s(%(name)s): %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, config):

        self.config = config
        if not hasattr(config, "spacy_model"):
            config.spacy_model = "en_core_web_trf"

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

    def tokenize(self, sentence: str) -> torch.Tensor:

        if not hasattr(self, "tokenizer"):
            self.setup_spacy()

        tokenized = self.tokenizer(sentence)
        return tokenized._.trf_data.tokens["input_ids"]


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()
    preprocessor = Preprocessor(data_args)

    document = "Spacy is fucking hard to use arghggh!"
    print(preprocessor.tokenize(document))
