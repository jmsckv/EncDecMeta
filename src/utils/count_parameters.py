from prettytable import PrettyTable
from utils.printing import fancy_print

def count_parameters(model, return_count=True, save_to_file:str='./parameter_count.txt'):
    # return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    table = PrettyTable(["Modules", "Parameters"])#
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    if save_to_file:
        with open(save_to_file, 'w') as f:
            m = fancy_print(f'Parameter count for sampled architecture {model.config["arch_string"]}', return_string=True)
            print(m, file=f)
            print(table, file=f)
            print(f"Total Trainable Params: {total_params}", file=f)
    if return_count:
        return total_params



