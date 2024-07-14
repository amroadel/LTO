import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        self.build_model(backbone)

    def build_model(self, backbone) -> None:
        self.backbone = backbone

    def forward(self, 
        x_spt: Tensor, 
        y_spt: Tensor, 
        x_qry: Tensor
    ) -> Tensor:
        """
        x_spt.shape = [n_spt, C, H, W]
        y_spt.shape = [n_spt]
        x.shape = [nQry, C, H, W]
        """
        num_classes = y_spt.max() + 1 # NOTE: assume B==1

        B, n_spt, C, H, W = x_spt.shape
        spt_feats = self.backbone(x_spt.view(-1, C, H, W))
        spt_feats = spt_feats.view(B, n_spt, -1)

        y_spt_1hot = F.one_hot(y_spt, num_classes).transpose(1, 2) # B, nC, n_spt

        # B, nC, n_spt x B, n_spt, d = B, nC, d
        prototypes = torch.bmm(y_spt_1hot.float(), spt_feats)
        prototypes = prototypes / y_spt_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images

        k_qry = x_qry.shape[1]
        qry_feats = self.backbone(x_qry.view(-1, C, H, W))
        qry_feats = qry_feats.view(B, k_qry, -1) # B, nQry, d

        logits = self.cos_classifier(prototypes, qry_feats) # B, nQry, nC
        return logits.view(B * k_qry, -1)

    def cos_classifier(self, 
        w: Tensor, 
        f: Tensor
    ) -> Tensor:
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=-1)
        w = F.normalize(w, p=2, dim=-1)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores