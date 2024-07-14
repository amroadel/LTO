from learners.base import LearnerFewShot
from learners.protonet.model import ProtoNet

class LearnerProtoNet(LearnerFewShot):
    name = 'protonet'

    def build_model(self, is_training=False, resample=False) -> None:
        self.model = ProtoNet(self.backbone)