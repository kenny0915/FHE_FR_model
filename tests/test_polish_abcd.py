import unittest
import torch
from torch import nn
from controlled_degree2.model import DirectQuadratic
from controlled_degree2.polish_abcd import PolishObjective

class Toy(nn.Module):
    def __init__(self,quadratic=True):
        super().__init__()
        self.conv=nn.Conv2d(2,2,1,bias=False)
        with torch.no_grad():self.conv.weight.copy_(torch.eye(2).reshape(2,2,1,1))
        self.bn=nn.BatchNorm2d(2)
        def act(name):
            return DirectQuadratic(2,lam_fit=1.,lam_reg=.6,slope=.2,
                     coeffs=torch.tensor([[0.,.6,.2],[0.,.6,.2]]),name=name) if quadratic else nn.PReLU(2,.2)
        self.prelu=act('prelu');self.layer3=nn.Module();self.layer3.prelu=act('layer3.prelu')
        self.head=nn.Linear(2,2)
    def forward(self,x):
        return self.head(self.layer3.prelu(self.prelu(self.bn(self.conv(x)))).mean((2,3)))

class PolishABCDTests(unittest.TestCase):
    def test_scopes_and_bn_modes(self):
        for arm in 'ABCD':
            obj=PolishObjective(Toy(),Toy(False),arm);obj.configure()
            self.assertEqual(obj.causal,['layer3.prelu'] if arm=='A' else ['prelu','layer3.prelu'])
            self.assertEqual(obj.student.bn.training,arm in 'AB')
            self.assertEqual(obj.student.prelu.clip,arm!='D')
    def test_d_routes_healthy_and_bad_rows_separately(self):
        obj=PolishObjective(Toy(),Toy(False),'D')
        x=torch.stack([torch.full((2,2,2),.2),torch.full((2,2,2),8.)])
        routes,_=obj.routes(x)
        self.assertEqual(routes.tolist(),[-1,0])
        before=obj.student.bn.running_mean.clone()
        loss=obj(x,torch.ones(2,dtype=torch.bool),.3,1.)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(obj.metrics['safe_rows'],1);self.assertEqual(obj.metrics['repair_rows'],1)
        self.assertTrue(torch.equal(before,obj.student.bn.running_mean))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters()))
        self.assertIsNone(obj.student.prelu.coeffs.grad)
    def test_prefix_stops_before_invalid_square(self):
        obj=PolishObjective(Toy(),Toy(False),'D');obj.configure()
        calls=[]
        h=obj.student.prelu.register_forward_hook(lambda *args:calls.append(1))
        try:
            loss=obj.repair(torch.full((2,2,2,2),1e20),0)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertEqual(calls,[])
        finally:h.remove()
    def test_c_fixes_moments_but_trains_affines(self):
        obj=PolishObjective(Toy(),Toy(False),'C');before=obj.student.bn.running_var.clone()
        loss=obj(torch.randn(4,2,2,2),torch.ones(4,dtype=torch.bool),.3,1.);loss.backward()
        self.assertTrue(torch.equal(before,obj.student.bn.running_var))
        self.assertIsNotNone(obj.student.bn.weight.grad)
    def test_all_pathological_rows_still_have_range_gradient(self):
        obj=PolishObjective(Toy(),Toy(False),'D')
        loss=obj(torch.ones(2,2,2,2)*8,torch.zeros(2,dtype=torch.bool),.3,1.);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(obj.metrics['repair_rows'],0)
        self.assertGreater(float(obj.student.conv.weight.grad.abs().sum()),0)

class ResultSummaryTests(unittest.TestCase):
    def test_nonfinite_result_is_not_accepted(self):
        import json,tempfile,sys
        from pathlib import Path
        from unittest.mock import patch
        from controlled_degree2.polish_abcd_result import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'ms1mv3').mkdir();(root/'ijbc/polish_abcd').mkdir(parents=True)
            (root/'ms1mv3/manifest.json').write_text(json.dumps({'output_nonfinite':[]}))
            a=dict(source_images=469375,augmented_embeddings=938750,audited_input_rows=938750,
                   audited_output_rows=938750,embedding_nonfinite_rows=0,nonfinite_values=1)
            (root/'ijbc/finite_audit.json').write_text(json.dumps(a))
            (root/'ijbc/polish_abcd/ijbc_tar_at_far_raw.json').write_text(json.dumps([{'points':{'0.0001':{'at_or_below_requested_far':{'tar_percent':99.,'actual_far':.0001}}}}]))
            with patch.object(sys,'argv',['summary','--root',tmp]):main()
            result=json.loads((root/'summary.json').read_text());self.assertIsNone(result['valid_tar_at_far_1e4'])
            a['nonfinite_values']=0;a['audited_output_rows']=1
            (root/'ijbc/finite_audit.json').write_text(json.dumps(a))
            with patch.object(sys,'argv',['summary','--root',tmp]):
                with self.assertRaises(ValueError):main()

if __name__=='__main__':unittest.main()
