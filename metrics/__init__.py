import os
from .evaluators import BaseEvaluator, FewShotEvaluator, AttributeEvaluator

def get_evaluator(cfg, datamodule) -> BaseEvaluator:
    evaluator_type = cfg.TYPE
    match evaluator_type:
        case 'classification':
            return FewShotEvaluator(
                datamodule.restricted_labels,
                datamodule.superclasses, 
                datamodule.superclass_names, 
                filename=os.path.join(
                    cfg.EVAL_DIR,
                    f'{datamodule.superclass_id}.tsv'))
        case 'attribute':
            return AttributeEvaluator(
                attributes=datamodule.attributes,
                filename=os.path.join(
                    cfg.EVAL_DIR,
                    f'{datamodule.superclass_id}.tsv'))
        case _:
            raise ValueError(evaluator_type)