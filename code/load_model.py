from models import Bert

MODEL_DICT = {"bert": Bert}


def load_model(model_args):

    model_name_or_path = model_args.model_name_or_path
    model = MODEL_DICT[model_name_or_path.lower()](model_args)
    return model
