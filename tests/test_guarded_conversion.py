import torch
from torch import nn

from controlled_degree2.guarded_conversion import GuardedConversion
from controlled_degree2.model import DirectQuadratic


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.prelu = DirectQuadratic(1, lam_fit=1., name='prelu')
        self.prelu.alpha = 1.
        self.prelu.clip = self.prelu.clip_eval = False
        self.bn1.eval()

    def forward(self, x):
        return self.prelu(self.bn1(x)).flatten(1)


def test_routing_repairs_overflow_input_and_preserves_good_feature_gradient():
    backbone = Toy()
    backbone.prelu.coeffs.requires_grad_(True)
    model = GuardedConversion(backbone, feature_dim=4)
    images = torch.stack([torch.full((1, 2, 2), .25), torch.full((1, 2, 2), 1e20)])
    features, rows, repair, scores = model(images)
    assert rows.tolist() == [0] and scores[1] > 4
    assert torch.isfinite(features).all() and torch.isfinite(repair)
    (features.square().mean()+repair).backward()
    assert torch.isfinite(backbone.bn1.weight.grad).all()
    assert backbone.prelu.coeffs.grad is not None
    assert torch.isfinite(backbone.prelu.coeffs.grad).all()
    assert not backbone.prelu.clip and not backbone.prelu.clip_eval
    assert not backbone.bn1.training and backbone.prelu.training


def test_all_escaping_rows_still_produce_repair_gradients():
    backbone = Toy()
    model = GuardedConversion(backbone, feature_dim=4)
    features, rows, repair, scores = model(torch.full((2, 1, 2, 2), 1e20))
    assert not len(rows) and features.shape == (1, 4)
    assert features.eq(0).all() and scores.gt(4).all()
    repair.backward()
    assert backbone.bn1.weight.grad is not None
    assert torch.isfinite(backbone.bn1.weight.grad).all()


def test_good_rows_are_identical_to_unclipped_backbone():
    backbone = Toy()
    model = GuardedConversion(backbone, feature_dim=4)
    images = torch.randn(3, 1, 2, 2)*.1
    expected = backbone(images)
    features, rows, repair, _ = model(images)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    assert rows.tolist() == [0, 1, 2] and repair == 0


def _mixed_rank_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from controlled_degree2.recipe_a import IdentityHead
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(1)
        backbone = Toy()
        backbone.prelu.coeffs.requires_grad_(True)
        model = DDP(GuardedConversion(backbone, feature_dim=4), find_unused_parameters=True,
                    broadcast_buffers=False)
        head = DDP(IdentityHead(torch.randn(3, 4)))
        optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.001)
        for step in range(3):
            # Swap the all-repair rank so parameter usage changes across steps.
            images = torch.full((2, 1, 2, 2), 1e20 if rank == step % 2 else .25)
            optimizer.zero_grad(set_to_none=True)
            features, rows, repair, _ = model(images)
            mask = torch.ones(len(features), dtype=torch.bool) if len(rows) else torch.zeros(1, dtype=torch.bool)
            loss = head(features, torch.zeros(len(features), dtype=torch.long), mask)+repair
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            flat = torch.cat([p.detach().flatten() for p in list(model.parameters())+list(head.parameters())])
            reference = flat.clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(flat, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ddp_mixed_full_and_all_repair_ranks_stay_synchronized(tmp_path):
    torch.multiprocessing.spawn(_mixed_rank_worker, args=(str(tmp_path/'rendezvous'),), nprocs=2)


def test_repair_rows_are_kept_in_training_replay():
    from controlled_degree2.recipe_a import TrainingTailReplay
    replay = TrainingTailReplay(capacity=2)
    images = torch.tensor([1., 20.]).reshape(2, 1, 1, 1)
    replay.update(images, torch.tensor([3, 7]), torch.ones(2, dtype=torch.bool), [],
                  scores=torch.tensor([1., 20.]))
    inserted, labels, mask = replay.inject(torch.zeros_like(images), torch.zeros(2, dtype=torch.long),
                                           torch.ones(2, dtype=torch.bool))
    assert inserted[0].item() == 20 and labels[0] == 7 and mask[0]
