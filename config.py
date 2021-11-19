from dataclasses import  dataclass, field


@dataclass
class MiscellaneousArguments:
    seed: int = field(default=42, metadata={"help": "Fixate seed."})


@dataclass
class DataArguments:
    data_dir: str = field(default="data/", metadata={"help": "Where data locates"})
    dataset_name: str = field(
        default="paperswithcode",
        metadata={"help": "The name of the dataset in arrow format."},
    )
    init_data_pct: float = field(
        default=0.05,
        metadata={
            "help": "How much data to use in the first place. Use 1.0 for naive training with a full dataset."
        },
    )

    balanced: bool = field(default=False, metadata={"help": ""})
    use_abstract: bool = field(default=False, metadata={"help": ""})
    use_task_id: bool = field(default=False, metadata={"help": ""})


@dataclass
class TrainingArguments:

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
