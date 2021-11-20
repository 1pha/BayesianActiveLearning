from dataclasses import dataclass, field


@dataclass
class DataArguments:

    asset_dir: str = field(
        default="../assets/",
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
        default="paperswithcode",
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
        default=128, metadata={"help": "Number of data per batch. Default=128."}
    )


@dataclass
class TrainerArguments:

    seed: int = field(default=42, metadata={"help": "Fixate seed."})
    model_name_or_path: str = field(
        default="cnnlstm",
        metadata={"help": "Selection of model. One of svm/cnnlstm/bert can be used."},
    )

    increment_pct: float = field(
        default=0.05, metadata={"help": "How much data to be incremented. (In %)"}
    )
    approximation: str = field(
        default="mcdropout",
        metadata={"help": "Which approximation to use for distribution of the "},
    )
    acquisition: str = field(
        default="lc",
        metadata={
            "help": "Which acquisition function to use. You can either use lc(Least Confidence)/mnlp/bald/batchbald"
        },
    )


@dataclass
class ModelArguments:

    max_seq_len: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length. Default value is 128. If you are using abstract, please use larger sequence length."
        },
    )
    embed_dim: int = field(
        default=768, metadata={"help": "Embedding dimension used inside the models."}
    )
    num_layers: int = field(
        default=6, metadata={"help": "Number of hidden layers for models."}
    )
    intermediate_size: int = field(
        default=768,
        metadata={
            "help": "Intermediate size of the vector that is used in feed-forward"
        },
    )
    dropout_proba: float = field(
        default=0.2,
        metadata={
            "help": "Dropout ratio used in the model. Use 20% for default and all same dropout ratio will be used in attention, feedforward and last classifier layer in transformer models."
        },
    )


def parse_arguments():

    from transformers import HfArgumentParser, set_seed

    parser = HfArgumentParser((DataArguments, TrainerArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Fixate Seed
    set_seed(training_args.seed)

    # Exchange configurations between argument set.
    data_args.seed = training_args.seed
    data_args.max_seq_len = model_args.max_seq_len

    # TODO Check dependency

    return data_args, training_args, model_args


if __name__ == "__main__":

    data_args, training_args, model_args = parse_arguments()
    print(data_args, training_args, model_args)
