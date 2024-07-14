import numpy as np
from torch import Tensor

from .accuracy import accuracy_superclass, auroc_attribute
from .meter import AverageMeter
from .report import Report

class BaseEvaluator:

    def __init__(self, filename: str):
        r'''
        '''
        self.data = []
        self.acc_meter = {}
        self.filename = filename

        self.outer_epoch = -1
        self.inner_epoch = -1
        self.last_inner_epoch = -1
        self.original_acc_r = None
        self.original_acc_o = None

    def __str__(self) -> str:
        return str(self.report)
    
    def reset(self) -> None:
        for key in self.keys:
            self.acc_meter[key].reset()

    def set_inner_epoch(self, epoch: int) -> None:
        self.inner_epoch = epoch

    def set_outer_epoch(self, epoch: int) -> None:
        self.outer_epoch = epoch

    def log(self) -> None:
        r'''Print the evaluation results and write them to `self.filename` in tsv
        '''
        print(self.report.stringfy(self.data))
        print(f'[*] Saving results to {self.filename}')
        self.report.save(self.data, self.filename)

    def update(self, 
        logits: Tensor, 
        targets: Tensor, 
        labels: Tensor | None = None
    ) -> None:
        r'''Update the accuracy meters
        '''
        if labels is None:
            labels = targets
        acc_list = accuracy_superclass(
            logits, targets, labels, self.superclasses, self.restricted_labels)
        for i, key in enumerate(self.keys):
            self.acc_meter[key].update(*acc_list[i])
        
    def eval(self) -> None:
        r'''Update the averages to the `report`
        '''
        values = [self.outer_epoch+1, self.inner_epoch+1] + [100*self.acc_meter[key].avg for key in self.keys ]
        values = self.add_drop_ratios(values)
        self.latest_values = values
        self.data.append(values)
        self.log()

    def add_drop_ratios(self,
        values: list,
    ) -> list:
        r'''
        - Args
        '''
        outer_epoch, inner_epoch, _, acc_r, acc_o, *_ = values
        self.last_inner_epoch = max(inner_epoch, self.last_inner_epoch)
        if outer_epoch == 0:
            self.original_acc_r = acc_r
            self.original_acc_o = acc_o
        if not self.data:
            ratio = 'N/A'        
        elif inner_epoch < self.last_inner_epoch:
            ratio = 'N/A'
        elif outer_epoch == 0:
            ratio = 'N/A'
        elif acc_o >= self.original_acc_o:
            ratio = (self.original_acc_r - acc_r)
            ratio = f'{round(ratio, 2)}*' # NOTE: no drop on acc others
        else:
            ratio = (self.original_acc_r - acc_r) / (self.original_acc_o - acc_o)
            ratio = round(ratio, 2)
        values.insert(2, ratio)
        return values
    
class FewShotEvaluator(BaseEvaluator):

    def __init__(self, 
        restricted_labels: Tensor, 
        superclasses: list[list[int]],
        superclass_names: list[str],
        filename: str
    ):
        r'''
        '''
        super().__init__(filename)
        assert(len(superclass_names) == len(superclasses))
        self.restricted_labels = restricted_labels
        self.superclasses = superclasses
        
        self.keys = ['all', 'restricted', 'other'] + superclass_names
        for key in self.keys:
            self.acc_meter[key] = AverageMeter()

        self.report = Report(
            headers=['Out', 'In', 'DropR', 'All', 'Res', 'Oth'] + list(range(len(superclass_names))),
            logidx=6)

    def update(self, 
        logits: Tensor, 
        targets: Tensor, 
        labels: Tensor | None = None
    ) -> None:
        r'''Update the accuracy meters
        '''
        if labels is None:
            labels = targets
        acc_list = accuracy_superclass(
            logits, targets, labels, self.superclasses, self.restricted_labels)
        for i, key in enumerate(self.keys):
            self.acc_meter[key].update(*acc_list[i])

class AttributeEvaluator(BaseEvaluator):

    def __init__(self, 
        attributes: list[str],
        superclass_id: int,
        filename: str
    ):
        r'''
        '''
        super().__init__(filename)
        self.superclass_id = superclass_id
        
        self.keys = ['all', 'restricted', 'other'] + attributes
        for key in self.keys:
            self.acc_meter[key] = AverageMeter()

        self.report = Report(
            headers=['Out', 'In', 'DropR', 'All', 'Res', 'Oth'] \
                + list(range(len(attributes))),
            logidx=6)

    def update(self, 
        logits: Tensor, 
        labels: Tensor
    ) -> None:
        r'''Update the accuracy meters
        '''
        eval_list = auroc_attribute(
            logits, labels, self.superclass_id)
        for i, key in enumerate(self.keys):
            self.acc_meter[key].update(*eval_list[i])