from tools import to_tensors

class InputProcessor:
    def process(self,data):
        # convert to tensors
        states,actions,rt_vals=to_tensors(*data)
        # TODO: devices 
        # types
        states=states.float()
        actions=actions.long()
        rt_vals=rt_vals.float()
        # TODO: normalization
        return states,actions,rt_vals