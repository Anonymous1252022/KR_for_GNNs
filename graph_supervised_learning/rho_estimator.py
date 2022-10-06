from typing import List, Type, Tuple
import pandas as pd
import torch.nn
from graph_supervised_learning.rho import compute_rho

class RhoEstimator:
    """
        Rho estimator class
    """
    def __init__(self, path: str, model: torch.nn.Module, layers_type: List[Type[torch.nn.Module]], rho_reg: float, lambda_: float, task_type: str):
        self._all_modules = self._get_all_modules(model, layers_type=tuple(layers_type))
        self._path = path # path to .csv file
        if self._path is not None:
            self._columns = ['iter'] +  ['layer_' + str(i) + '_rho' for i in range(len(self._all_modules))]
            pd.DataFrame(columns=self._columns).to_csv(self._path, index=False)

        self._rho_list = []

        self._last_layer = self._all_modules[-1]
        self._iter = 0
        self._y = None
        self._mask = None # we should select only training node representations
        self._rho = None
        self._rho_reg = rho_reg
        self._lambda = lambda_
        self._task_type=task_type

        for curr_module in self._all_modules:
            def forward_hook(module, input, output):
                if module.training == True:
                    self._rho_list.append(compute_rho(output[self._mask], self._y[self._mask], lambda_=self._lambda))

                    if module == self._last_layer:
                        self._rho = torch.stack(self._rho_list)[:-1].sum() # discard last layer
                        rho_list = [rho.item() for rho in self._rho_list]
                        data = [self._iter] + rho_list
                        if self._path is not None:
                            pd.DataFrame(data=[data], columns=self._columns).to_csv(
                                self._path, mode='a', index=False, header=False
                            )

                        self._iter += 1
                        self._rho_list = []

            curr_module.register_forward_hook(hook=forward_hook)

    @staticmethod
    def _get_all_modules(module: torch.nn.Module, layers_type: Tuple[Type[torch.nn.Module]]):
        ans = []
        if isinstance(module, layers_type):
            ans.append(module)

        m_childrens = list(module.children())
        if len(m_childrens) == 0:
            return ans
        else:
            for curr_children in m_childrens:
                ans += RhoEstimator._get_all_modules(curr_children, layers_type)
            return ans

    def set_y(self, y:torch.Tensor):
        if self._task_type == 's':
            self._y = torch.nn.functional.one_hot(y.flatten()).type(torch.float32)
            self._y = self._y[:, self._y.sum(0) != 0]  # prevents nans in back propagation
        else: # (m)
            self._y = y.type(torch.float32)

    def set_mask(self, mask:torch.Tensor):
        self._mask = mask

    @property
    def rho_reg(self):
        return self._rho_reg

    @property
    def rho(self):
        return self._rho
