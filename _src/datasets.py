import re
import itertools
import math
import torch
from torch import Tensor, LongTensor
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional


SPECIAL_TOKENS = [
    '<OP>', '+', '*', '-', '=', '<SOS>', '<EOS>', '<MOD>', '<MASK>', ',',
]

POSHINTS = ['<a>', '<b>', '<c>', '<d>', '<e>', '<f>', '<g>']


"""Helper functions"""

def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    else:
        return str(operand)

class BaseTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""

    def __init__(self, modulo: int, base: int) -> None:
        self.modulo = modulo
        self.base = base
        self.max_digits = int(math.log(modulo - 1, base)) + 1 if modulo - 1 > 0 else 1
        assert self.base <= self.modulo
        self.itos = self.get_tokens(self.base)
        self.stoi = dict([(s, i) for i, s in enumerate(self.itos)])

    def _int_to_base(self, num: int, base: int) -> str:
        num = int(num)
        if num == 0:
            return ','.join('0' for _ in range(self.max_digits))
        digits = []
        while num:
            digits.append(int(num % base))
            num //= base
        if len(digits) < self.max_digits:
            padding_length = self.max_digits - len(digits)
            digits.extend([0] * padding_length)
        return ','.join(str(x) for x in digits[::-1])

    def split_equation(self, equation: str, reverse_target = False, pos_hint = False) -> list:
        """
        Split special operators only '32' -> '28'
        Split all number to digits '32' -> '28' (base 12) -> ['2', '8']
        """
        def convert_and_split(match):
            num = int(match.group())
            converted = self._int_to_base(num, self.base)
            if pos_hint is False:
                return ''.join(converted)
            else:
                parts = converted.split(',')
                tagged_digits = []
                for i, digit in enumerate(parts):
                    tagged_digits.extend([POSHINTS[i], digit])
                return ''.join(tagged_digits)
            
        converted_equation = re.sub(r'\d+', convert_and_split, equation)
        splited_eq = re.findall(r'<\w+>|\d+|\*|\+|-|\/|=|<MOD>|<SOS>|<EOS>|<MASK>|<OP>|\S|,', converted_equation)


        splited_eq = [part for part in splited_eq if part != ',']
        
        if reverse_target == True:
            start_index = splited_eq.index('=')
            end_index = splited_eq.index('<EOS>')
            
            # Pair up tags and digits, reverse the list of pairs, and then flatten
            if pos_hint == True:
                pairs = [(splited_eq[i], splited_eq[i + 1]) for i in range(start_index + 1, end_index, 2)]
                reversed_pairs = pairs[::-1]
                reversed_sublist = [item for pair in reversed_pairs for item in pair]
            else:
                reversed_sublist = splited_eq[start_index + 1:end_index][::-1]

            splited_eq[start_index + 1:end_index] = reversed_sublist
        return splited_eq
    
    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s])

    def encode(self, eqs: List, reverse_target = False, pos_hint = False, show_seos = False) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        encoded_eqs = [self._encode(self.split_equation(s, reverse_target = reverse_target, pos_hint = pos_hint)) for s in eqs]
        return torch.stack(encoded_eqs, dim=0) if show_seos == True else torch.stack(encoded_eqs, dim=0)[:, 1:-1]
        
    def decode(self, tensor: Tensor) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()

        tokens = [self.itos[i] for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    @classmethod
    def get_tokens(cls, base):
        BASES = list(range(base + 1))
        tokens = list(map(render, BASES)) + list(map(render, SPECIAL_TOKENS)) + list(map(render, POSHINTS)) # Special Tokens
        tokens = tuple(tokens)
        return tokens


class BaseArithmeticDataset:

    def __init__(self, name: str, data: Union[Tensor, List[str]], modulo: int, base: Optional[int], train: bool, reverse_target: bool = False, pos_hint: bool = False, show_seos: bool = False) -> None:
        """
        Description: Creates a dataset given equations 

        Parameters: 
            - data: A list of equation strings. Each equation must have an '=' in it
        """
        self.modulo = modulo
        self.base = base
        self.tokenizer = BaseTokenizer(self.modulo, self.base)

        self.name = name
        self.train = train
                
        self.data = self.tokenizer.encode(data, reverse_target = reverse_target, pos_hint = pos_hint, show_seos = show_seos)
        return

    @classmethod
    def splits(cls, data_pct: float, task_pct: float, modulo: int, base: int, Ws: list, seed: int = 0, encrypted: bool = False, show_mod: bool = False, show_seos: bool = False, pos_hint: bool = False, reverse_target: bool = False, split_tasks: bool = True, split_data: bool = False) -> Tuple:
        """
        Description: Creates training and validation datasets
        
        Parameters: 
            - train_pct: percentage of data used for training data
            - operator: The arithmetic operator for the dataset; should be in the OPERATOR_LIST
        Returns:
            - Tuple (train_dataset, validation_dataset)
        """
        assert (data_pct > 0.0 and data_pct < 100.0)
        assert (task_pct > 0.0 and task_pct < 100.0)
        
        eqs = cls.make_data(modulo, Ws, encrypted = encrypted, show_mod = show_mod)
        indices = np.array(list(range( len(Ws) * ( modulo**len(Ws[0]) ) ))).reshape(len(Ws), -1) # (n_tasks, n_data)
        if split_tasks is False and split_data is False:
            raise Exception("Must split task or data, or both!")
        elif split_data is True:
            np.random.seed(seed)
            transposed_indices = indices.T
            np.random.shuffle(transposed_indices) # Shuffle along n_data
            indices = transposed_indices.T # (n_tasks, n_data), shuffled along -1 dim.
            data_columns = round(indices.shape[-1] * (data_pct / 100.0))
            if split_tasks is False:
                train_indices = indices[:, :data_columns].reshape(-1)
                test_indices = indices[:, data_columns:].reshape(-1)
                ds_name = f'tasks{len(Ws)}_dim{len(Ws[0])}_mod{modulo}_stask{split_tasks}_sdata{split_data}_datapct{data_pct:.1f}'
            elif split_tasks is True:
                task_rows = round(indices.shape[0] * (task_pct / 100.0))
                np.random.shuffle(indices) # shuffle along n_tasks
                train_indices = indices[:task_rows, :data_columns].reshape(-1)
                test_indices = indices[task_rows:, data_columns:].reshape(-1)
                ds_name = f'tasks{len(Ws)}_dim{len(Ws[0])}_mod{modulo}_stask{split_tasks}_taskpct{task_pct:.1f}_sdata{split_data}_datapct{data_pct:.1f}'
        elif split_tasks is True and split_data is False:
            np.random.seed(seed)
            task_rows = round(indices.shape[0] * (task_pct / 100.0))
            np.random.shuffle(indices) # shuffle along n_tasks
            train_indices = indices[:task_rows, :].reshape(-1)
            test_indices = indices[task_rows:, :].reshape(-1)
            ds_name = f'tasks{len(Ws)}_dim{len(Ws[0])}_mod{modulo}_stask{split_tasks}_taskpct{task_pct:.1f}_sdata{split_data}'
        
        train_ds = cls(ds_name, np.array(eqs)[train_indices].tolist(), modulo, base, train = True, reverse_target = reverse_target, pos_hint = pos_hint, show_seos = show_seos)
        val_ds = cls(ds_name, np.array(eqs)[test_indices].tolist(), modulo, base, train = False, reverse_target = reverse_target, pos_hint = pos_hint, show_seos = show_seos)
        
        return train_ds, val_ds

    @classmethod
    def make_data(cls, modulo: int, Ws: list, encrypted = True, show_mod = False) -> List[str]:
        data = cls._make_equations(modulo, Ws, encrypted = encrypted, show_mod = show_mod)
        return data
    
    @classmethod
    def _make_equations(cls, modulo: int, Ws: list[np.ndarray], encrypted = True, show_mod = False) -> List[str]:
        elems = list(range(modulo))
        Xs = list(itertools.product(elems, repeat=len(Ws[0])))
        eqs = list()
        
        assert modulo > 0
        
        if encrypted:
            for W in Ws:
                for X in Xs:
                    target = np.vdot(W, X) % modulo
                    eq_parts = [f'{X[i]}' for i in range(len(X))]
                    eq = ' '.join(eq_parts)
                    # print(eq)
                    if show_mod == False:
                        eqs.append(f'<SOS> {eq} {target} <EOS>')
                    elif show_mod == True:
                        eqs.append(f'<SOS> {eq} <MOD> {modulo} {target} <EOS>')
        else:
            for W in Ws:
                for X in Xs:
                    target = np.vdot(W, X) % modulo
                    eq_parts = [f'{W[i]} * {X[i]}' for i in range(len(X))]
                    eq = ' + '.join(eq_parts)
                    # print(eq)
                    if show_mod == False:
                        eqs.append(f'<SOS> {eq} {target} <EOS>')
                    elif show_mod == True:
                        eqs.append(f'<SOS> {eq} <MOD> {modulo} {target} <EOS>')

        return eqs


def prepare_data(args, Ws) -> Tuple:
    data = {'datasets': BaseArithmeticDataset.splits(data_pct=args.data_pct, task_pct=args.task_pct, modulo=args.p, base=args.base, Ws=Ws, seed=args.data_seed, encrypted=args.encrypted, show_mod=args.show_mod, show_seos=args.show_seos, pos_hint=args.pos_hint, reverse_target=args.reverse_target, split_tasks=args.split_tasks, split_data=args.split_data), 'Ws': Ws, 'args': args}
    train_ds, val_ds = data['datasets']
    return train_ds.data, val_ds.data, train_ds.tokenizer


########################################################################################################################
###################################### Extra Code for inference ########################################################
########################################################################################################################
class BaseArithmeticDatasetGrid(BaseArithmeticDataset):
    
    def __init__(self, name: str, data: Union[Tensor, List[str]], modulo: int, base: Optional[int], train: bool, reverse_target: bool = False, pos_hint: bool = False, show_seos: bool = False) -> None:
        super().__init__(name, data, modulo, base, train, reverse_target, pos_hint, show_seos)
        
        self.modulo = modulo
        self.base = base
        self.tokenizer = BaseTokenizer(self.modulo, self.base)

        self.name = name
        self.train = train
                
        self.data = self.tokenizer.encode(data, reverse_target = reverse_target, pos_hint = pos_hint, show_seos = show_seos)
        return
        
    @classmethod
    def splits(cls, task_pct: float, modulo: int, base: int, Ws: list, seed: int = 0, encrypted: bool = False, show_mod: bool = False, show_seos: bool = False, pos_hint: bool = False, reverse_target: bool = False) -> Tuple:
        """
        Description: Creates training and validation datasets
        
        Parameters: 
            - train_pct: percentage of data used for training data
            - operator: The arithmetic operator for the dataset; should be in the OPERATOR_LIST
        Returns:
            - Tuple (train_dataset, validation_dataset)
        """
        # assert (data_pct > 0.0 and data_pct < 100.0)
        assert (task_pct > 0.0 and task_pct < 100.0)
        
        eqs = cls.make_data(modulo, Ws, encrypted = encrypted, show_mod = show_mod)
        indices = np.array(list(range( len(Ws) * ( modulo**len(Ws[0]) ) ))).reshape(len(Ws), -1) # (n_tasks, n_data)

        np.random.seed(seed)
        grid_indices = indices[:, :].reshape(-1)
        ds_name = f'tasks{len(Ws)}_dim{len(Ws[0])}_mod{modulo}'
        
        grid_ds = cls(ds_name, np.array(eqs)[grid_indices].tolist(), modulo, base, train = True, reverse_target = reverse_target, pos_hint = pos_hint, show_seos = show_seos)
        
        return grid_ds
    

def prepare_data_grid(args, Ws) -> Tuple:
    data = {'datasets': BaseArithmeticDatasetGrid.splits(task_pct=args.task_pct, modulo=args.p, base=args.base, Ws=Ws, seed=args.data_seed, encrypted=args.encrypted, show_mod=args.show_mod, show_seos=args.show_seos, pos_hint=args.pos_hint, reverse_target=args.reverse_target), 'Ws': Ws, 'args': args}
          
    grid_ds = data['datasets']
    return grid_ds.data, grid_ds.tokenizer