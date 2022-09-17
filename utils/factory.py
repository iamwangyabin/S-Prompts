from methods.sprompt import SPrompts

def get_model(model_name, args):
    name = model_name.lower()
    options = {'sprompts': SPrompts,
               }
    return options[name](args)

