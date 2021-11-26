from models import Bert, CNN_MC, BiLSTM_MC

MODEL_DICT = {"bert": Bert, "bilstm": BiLSTM_MC, "cnnlstm": CNN_MC}


def load_model(model_args):

    model_name_or_path = model_args.model_name_or_path
    model = MODEL_DICT[model_name_or_path.lower()](model_args)
    return model


def _load_base_config(training_args, model_args):

    model_name = model_args.model_name_or_path
    model_args.embed_dim = 256
    if model_name == "bert":
        model_args.num_layers = 4
        model_args.intermediate_size = 768

    elif model_name == "cnnlstm":
        model_args.out_channels = 512
        training_args.learning_rate = 0.001
        training_args.weight_decay = 0

    elif model_name == "bilstm":
        model_args.intermediate_size = 256
        training_args.learning_rate = 0.001
        training_args.weight_decay = 0

    return training_args, model_args


def _load_large_config(training_args, model_args):

    model_name = model_args.model_name_or_path
    if model_name == "bert":
        model_args.num_layers = 6
        model_args.intermediate_size = 3072

    elif model_name == "cnnlstm":
        model_args.out_channels = 1024

    elif model_name == "bilstm":
        model_args.intermediate_size = 512

    return training_args, model_args


def _num_params(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()

    model = load_model(model_args)
    print(model)
    print(_num_params(model))
