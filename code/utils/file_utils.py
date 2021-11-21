import os
import json
import logging
from pathlib import Path
from datetime import datetime as dt

import torch

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_today():

    td = dt.today()
    month_map = {
        11: "Nov ",
        12: "Dec ",
    }

    return (
        month_map[td.month]
        + str(td.day).zfill(2)
        + " "
        + str(td.hour).zfill(2)
        + ":"
        + str(td.minute).zfill(2)
        + " "
    )


def check_dir(config):

    output_dir = config.output_dir
    overwrite_flag = config.overwrite_output_dir

    if not overwrite_flag:
        # Don't overwrite
        n = 0
        output_dir += f"_({n})"
        while os.path.exists(output_dir):
            # Loop will break when output_dir_(n) does not exist
            n += 1
            output_dir += f"_({n})"
        return output_dir
    else:
        # Fine to overwrite
        return output_dir


def save_state(model, step, config):

    """config: TrainingArguments"""

    output_dir = check_dir(config)

    fname = Path(f"{output_dir}/{str(step).zfill(4)}.pt")
    torch.save(model.state_dict(), fname)


def check_name(mapper, arg):

    if arg in mapper:
        return mapper[arg]

    else:
        logger.warn(f"No {arg} was found in mapper. Can you check?: {arg}")
        raise


def generate_runname_tags(data_args, training_args, model_args):

    with open(
        Path(f"{data_args.asset_dir}/{training_args.configuration_keys}"), "r"
    ) as f:
        configuration_keys = json.load(f)

    init_pct = f"INIT{data_args.init_pct * 100}%"
    model_name = model_args.model_name_or_path.capitalize()
    active_learning = "Naive" if not training_args.active_learning else "AL"  # boolean
    if training_args.active_learning:
        approximation = check_name(
            configuration_keys["approximation"], training_args.approximation
        )
        acquisition = check_name(
            configuration_keys["acquisition"], training_args.acquisition
        )

        tags = [model_name, approximation, acquisition, init_pct, active_learning]

    else:

        tags = [model_name, init_pct, active_learning]

    run_name = get_today()
    run_name += " ".join(tags)
    return run_name, tags


if __name__ == "__main__":

    import sys

    sys.path.append("..")
    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()
    print(generate_runname_tags(training_args, model_args))
