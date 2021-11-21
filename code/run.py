import logging

import wandb

from config import parse_arguments, get_wandb_config, save_config
from dataset import build_dataset, build_dataloader
from trainer import NaiveTrainer, ActiveTrainer
from utils.file_utils import generate_runname_tags, check_dir

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("main.py")


def main():

    data_args, training_args, model_args = parse_arguments()
    run_name, tags = generate_runname_tags(data_args, training_args, model_args)
    wandb.init(
        project=training_args.project,
        tags=tags,
        name=run_name,
        config=get_wandb_config(data_args, training_args, model_args),
    )
    training_args.output_dir = check_dir(training_args)
    logger.info(f"Your results will be saved in: {training_args.output_dir}")
    logger.info(f"Start Training.")

    # 1. Initialization
    #   a. Initialize Dataset
    if training_args.do_train:
        dataset, model_args.vocab_size, model_args.num_labels = build_dataset(
            data_args, "train"
        )
        train_dataloader = build_dataloader(
            dataset,
            data_args,
        )

    if training_args.do_valid:
        dataset, model_args.vocab_size, model_args.num_labels = build_dataset(
            data_args, "valid"
        )
        valid_dataloader = build_dataloader(
            dataset,
            data_args,
        )

    if training_args.do_test:
        dataset, model_args.vocab_size, model_args.num_labels = build_dataset(
            data_args, "test"
        )
        test_dataloader = build_dataloader(
            dataset,
            data_args,
        )

    #   b. Initialize Trainer
    trainer = (
        NaiveTrainer(
            training_args,
            model_args,
            training_dataset=train_dataloader if training_args.do_train else None,
            validation_dataset=valid_dataloader if training_args.do_valid else None,
            test_dataset=test_dataloader if training_args.do_test else None,
        )
        if not training_args.active_learning
        else ActiveTrainer(
            training_args,
            model_args,
            training_dataset=train_dataloader if training_args.do_train else None,
            validation_dataset=valid_dataloader if training_args.do_valid else None,
            test_dataset=test_dataloader if training_args.do_test else None,
        )
    )
    trainer.run()
    save_config(
        training_args.output_dir,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
    )


if __name__ == "__main__":
    main()
