from methods.sprompt import SPrompts
from methods.coop import coop_ganfake

def get_model(model_name, args):
    name = model_name.lower()
    options = {'sprompts': SPrompts,
               'orgcoop': coop_ganfake,
               }
    return options[name](args)

