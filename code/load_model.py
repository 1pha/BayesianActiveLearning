from models import Bert, CNN_MC, BiLSTM_MC

MODEL_DICT = {"bert": Bert, "bilstm": BiLSTM_MC, "cnnlstm": CNN_MC}


def load_model(model_args):

    model_name_or_path = model_args.model_name_or_path
    model = MODEL_DICT[model_name_or_path.lower()](model_args)
    return model


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()

    print(load_model(model_args))
