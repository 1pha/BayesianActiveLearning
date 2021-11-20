from dataclasses import dataclass, field


@dataclass
class DataArguments:
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
    use_abstract: bool = field(default=False, metadata={"help": ""})
    use_task_id: bool = field(default=False, metadata={"help": ""})
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
    embeeding_dim: int = field(
        default=768, metadata={"help": "Embedding dimension used inside the models."}
    )


def parse_arguments():

    from transformers import HfArgumentParser, set_seed

    parser = HfArgumentParser((DataArguments, TrainerArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    data_args.seed = training_args.seed
    return data_args, training_args, model_args


if __name__ == "__main__":

    data_args, training_args, model_args = parse_arguments()
    print(data_args, training_args, model_args)
