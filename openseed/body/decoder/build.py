from .registry import model_entrypoints
from .registry import is_model


def build_decoder(config, *args, **kwargs):
    model_name = config['MODEL']['DECODER']['NAME'] #openseed_decoder

    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)