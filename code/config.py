import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class BaseArguments:
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def __repr__(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def load_config(self, json_file, arg_name=None):

        msg = "Load Configuration"
        msg += "" if arg_name is None else f" {arg_name}"
        logger.info(msg)

        try:
            for k, v in json_file.items():
                setattr(self, k, v)
        except:
            logger.warn("Failed to Load Configuration")
            raise


@dataclass
class DataArguments(BaseArguments):

    asset_dir: str = field(
        default="./assets/",
        metadata={
            "help": "Directory of assets where mapper and model checkpoints are saved."
        },
    )
    area2idx: str = field(
        default="area2idx.json",
        metadata={"help": "Name of the area-index mapper json file."},
    )
    data_dir: str = field(default="data/", metadata={"help": "Where data locates"})
    dataset_name: str = field(
        default="paperswithcode_balanced_tokenized",  # "paperswithcode",
        metadata={"help": "The name of the dataset in arrow format."},
    )
    init_pct: float = field(
        default=0.05,
        metadata={
            "help": "How much data to use in the first place. Use 1.0 for naive training with a full dataset."
        },
    )
    balanced: bool = field(default=False, metadata={"help": ""})
    use_abstract: bool = field(
        default=False,
        metadata={
            "help": "Whether to use abstract or not. Default to False (not using abstract)."
        },
    )
    use_task_id: bool = field(
        default=False,
        metadata={
            "help": "Whether to use task_id or not. Default to False (not using task_id)."
        },
    )
    preprocessor: str = field(
        default="spacy",
        metadata={
            "help": "Which preprocessing modules should be used. You can use one of spacy/nltk/huggingface tokenizer."
        },
    )
    spacy_model: str = field(
        default="en_core_web_trf",
        metadata={"help": "Choose spacy model. Please use en_core_web_trf only."},
    )

    batch_size: int = field(
        default=256, metadata={"help": "Number of data per batch. Default=128."}
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Setting this true will speed-up transferring data to cuda."},
    )


@dataclass
class TrainerArguments(BaseArguments):

    seed: int = field(default=42, metadata={"help": "Fixate seed."})
    configuration_keys: str = field(
        default="configuration_keys.json",
        metadata={"help": "Json file of the configuration key mapper."},
    )
    active_learning: bool = field(
        default=False,
        metadata={
            "help": "Whether to use active learning or not. If set to False, some configurations might change in order to train in naive way."
        },
    )
    increment_pct: float = field(
        default=0.05,
        metadata={
            "help": "How much data to be incremented. (In %) This is no longer used. Please use `increment_num` to control number of data to be acquired every epoch."
        },
    )
    increment_num: int = field(
        default=300, metadata={"help": "Increment amount from pooled data."}
    )
    acquisition_period: int = field(
        default=1,
        metadata={"help": "Period of acquiring extra data to training dataset."},
    )
    approximation: str = field(
        default="mcdropout",
        metadata={
            "help": "Which approximation to use for distribution of the model weights. Use one of single/mc/ensemble"
        },
    )
    num_sampling: int = field(
        default=5,
        metadata={
            "help": "Number of models to be used when approximating with MC Dropout and Full Ensemble."
        },
    )
    acquisition: str = field(
        default="lc",
        metadata={
            "help": "Which acquisition function to use. You can either use lc(Least Confidence)/mnlp/bald/batchbald"
        },
    )
    save_confidence: bool = field(
        default=True, metadata={"help": "Whehter to save confidence level or not."}
    )
    output_dir: str = field(
        default="output/default", metadata={"help": "Saving directory. Must be given."}
    )
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "If True, overwrite to directory"}
    )
    checkpoint_period: int = field(
        default=5, metadata={"help": "Period of checkpoints to save."}
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to train or not."})
    do_valid: bool = field(
        default=False, metadata={"help": "Whether to test metric with validation data."}
    )
    do_test: bool = field(
        default=False,
        metadata={"help": "Whether to test metric with test data. No peeking!"},
    )
    num_train_epochs: int = field(
        default=100, metadata={"help": "Number of epochs to train."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps."}
    )
    optimizer: str = field(
        default="adamw",
        metadata={
            "help": "Which optimizer to use. If Transformer-family is selected, AdamW will be used, and adam will be used if one of recurrent-family is chosen as a model."
        },
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "Learning Rate. Defaults to 1e-4."}
    )
    warmup_steps: int = field(
        default=100, metadata={"help": "Warmup steps for transformer models."}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Decoupled weight decay to apply in AdamW."}
    )
    fp16: bool = field(
        default=True,
        metadata={
            "help": "Whether to use mixed precision. Setting this true will boost training speed."
        },
    )
    use_gpu: bool = field(
        default=True,
        metadata={
            "help": "Whether to use gpu or not. If set to False, it will force the device not to use gpu even though you have it."
        },
    )
    project: str = field(
        default="Active-Learning", metadata={"help": "Name of the wandb project name."}
    )
    run_name: str = field(
        default=None,
        metadata={
            "help": "Run name to be displayed on wandb. If None is given, it will generate a run_name with a certain pattern."
        },
    )
    FORCE_RUN_NAME: str = field(
        default=False,
        metadata={
            "help": "Force run name to be other name. If True given, use TEST_RUN as run name for wandb. If string given, use that string as a run name and default is False - use generate_run_name to name it."
        },
    )


@dataclass
class ModelArguments(BaseArguments):

    model_name_or_path: str = field(
        default="bert",
        metadata={"help": "Selection of model. One of svm/cnnlstm/bert can be used."},
    )
    vocab_size: int = field(
        default=83931, metadata={"help": "Number of vocabulary size in preprocessor."}
    )
    max_seq_len: int = field(
        default=64,
        metadata={
            "help": "Maximum sequence length. Default value is 128. If you are using abstract, please use larger sequence length."
        },
    )
    embed_dim: int = field(
        default=256, metadata={"help": "Embedding dimension used inside the models."}
    )
    num_layers: int = field(
        default=4, metadata={"help": "Number of hidden layers for models."}
    )
    num_attention_heads: int = field(
        default=8,
        metadata={
            "help": "Number of attention heads. Default to 8, bert-base uses 12."
        },
    )
    intermediate_size: int = field(
        default=768,
        metadata={
            "help": "Intermediate size of the vector that is used in feed-forward"
        },
    )
    dropout_prob: float = field(
        default=0.2,
        metadata={
            "help": "Dropout ratio used in the model. Use 20% for default and all same dropout ratio will be used in attention, feedforward and last classifier layer in transformer models."
        },
    )
    num_labels: int = field(default=16, metadata={"help": "Number of total labels."})
    rnn_cell: str = field(
        default="lstm",
        metadata={"help": "Choose LSTm or GRU to for recurrent family models."},
    )
    bidirectional: bool = field(
        default=True,
        metadata={
            "help": "Wheter to use bidirectional encoder for recurrent family models. Default is True."
        },
    )
    out_channels: int = field(
        default=256,
        metadata={"help": "Number of channels to be used in CNN-LSTM layer."},
    )


def save_config(output_dir, **kwargs):

    logger.info("Parse Arguments into dict.")
    arguments = dict()
    for arg_name, args in kwargs.items():
        arguments[arg_name] = args.to_dict()

    with open(Path(f"{output_dir}/config.json"), "w") as f:
        json.dump(arguments, f)


def load_config(output_dir):

    json_fname = Path(f"{output_dir}/config.json")
    with open(json_fname, "r") as f:
        return json.load(f)

    # TODO needs revision


def get_wandb_config(data_args, training_args, model_args):

    config = dict(**model_args.to_dict())
    config.update(
        {
            # Training Arguments
            "active_learning": training_args.active_learning,
            "increment_num": training_args.increment_num,
            "approximation": training_args.approximation,
            "acquisition": training_args.acquisition,
            "output_dir": training_args.output_dir,
            "overwrite_output_dir": training_args.overwrite_output_dir,
            "do_train": training_args.do_train,
            "do_valid": training_args.do_valid,
            "do_test": training_args.do_test,
            "num_train_epochs": training_args.num_train_epochs,
            "optimizer": training_args.optimizer,
            "learning_rate": training_args.learning_rate,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
            # Data Arguments
            "init_pct": data_args.init_pct,
            "balanced": data_args.balanced,
            "use_abstract": data_args.use_abstract,
            "use_task_id": data_args.use_task_id,
            "batch_size": data_args.batch_size,
        }
    )
    return config


def parse_arguments():

    import torch
    from transformers import HfArgumentParser, set_seed

    parser = HfArgumentParser((DataArguments, TrainerArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Fixate Seed
    set_seed(training_args.seed)

    # Exchange configurations between argument set.
    data_args.seed = training_args.seed
    data_args.max_seq_len = model_args.max_seq_len

    # TODO Check dependency XXX For now, please manually choose init_pct!
    # if not training_args.active_learning:
    #     data_args.init_pct = 1.0

    if training_args.use_gpu and torch.cuda.is_available():
        pass

    else:
        training_args.use_gpu = False

    if training_args.acquisition in ["random", "lc", "margin", "entropy"]:
        training_args.approximation = "single"
        training_args.num_sampling = 1

    model_args.model_name_or_path = model_args.model_name_or_path.lower()

    return data_args, training_args, model_args


if __name__ == "__main__":

    data_args, training_args, model_args = parse_arguments()
    print(get_wandb_config(data_args, training_args, model_args))
