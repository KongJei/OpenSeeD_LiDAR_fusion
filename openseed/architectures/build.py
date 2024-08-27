from .registry import model_entrypoints
from .registry import is_model
# from .registry import _model_entrypoints

def build_model(config, **kwargs):
    model_name = config['MODEL']['NAME'] # openseed_model

    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')
    # print(_model_entrypoints)
    return model_entrypoints(model_name)(config, **kwargs)