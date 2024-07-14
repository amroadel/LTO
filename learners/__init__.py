from learners.base import LearnerBase
from learners.ce import LearnerCE, LearnerAttrCE
from learners.coop import LearnerCoop
from learners.tipadapter import LearnerTipAdapter
from learners.protonet import LearnerProtoNet

def get_learner(
    cfg, 
    backbone, 
    datamodule, 
    evaluator=None
):
    learner_type = cfg.TYPE
    match learner_type:
        case 'base':
            return LearnerBase(cfg, datamodule, evaluator)
        case 'ce':
            return LearnerCE(cfg, backbone, datamodule, evaluator)
        case 'ce_attr':
            return LearnerAttrCE(cfg, backbone, datamodule, evaluator)
        case 'coop':
            return LearnerCoop(cfg, backbone, datamodule, evaluator)
        case 'tipadapter':
            return LearnerTipAdapter(cfg, backbone, datamodule, evaluator)
        case 'protonet':
            return LearnerProtoNet(cfg, backbone, datamodule, evaluator)
        case _:
            raise ValueError(learner_type)