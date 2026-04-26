"""Test suite for motion-aware TPSGG.

Tests:
    - vIoU computation
    - R@K and mR@K metrics
    - Sinkhorn optimal transport distance
    - Positive/negative sampling
    - Synthetic dataloader
    - One training step
"""

import pytest
import torch
import numpy as np

# ==================== Transport Tests ====================

class TestOptimalTransport:
    def test_cosine_distance_matrix_shape(self):
        from motion_aware_tpsgg.transport import cosine_distance_matrix
        h_i = torch.randn(4, 16)
        h_j = torch.randn(6, 16)
        C = cosine_distance_matrix(h_i, h_j)
        assert C.shape == (4, 6)

    def test_cosine_distance_range(self):
        from motion_aware_tpsgg.transport import cosine_distance_matrix
        h_i = torch.randn(5, 32)
        h_j = torch.randn(5, 32)
        C = cosine_distance_matrix(h_i, h_j)
        assert (C >= -0.01).all(), "Cosine distance should be >= 0"
        assert (C <= 2.01).all(), "Cosine distance should be <= 2"

    def test_self_distance_zero(self):
        from motion_aware_tpsgg.transport import cosine_distance_matrix
        h = torch.randn(4, 16)
        C = cosine_distance_matrix(h, h)
        diag = torch.diagonal(C)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_sinkhorn_returns_scalar(self):
        from motion_aware_tpsgg.transport import sinkhorn_ot_distance, OTConfig
        h_i = torch.randn(4, 16)
        h_j = torch.randn(6, 16)
        config = OTConfig(n_iter=50, tau=0.1)
        d = sinkhorn_ot_distance(h_i, h_j, config)
        assert d.dim() == 0, "Should return scalar"

    def test_sinkhorn_nonnegative(self):
        from motion_aware_tpsgg.transport import sinkhorn_ot_distance, OTConfig
        h_i = torch.randn(3, 8)
        h_j = torch.randn(5, 8)
        config = OTConfig(n_iter=50, tau=0.1)
        d = sinkhorn_ot_distance(h_i, h_j, config)
        assert d.item() >= -0.01, "OT distance should be non-negative"

    def test_ot_similarity(self):
        from motion_aware_tpsgg.transport import ot_similarity, OTConfig
        h_i = torch.randn(4, 16)
        h_j = torch.randn(4, 16)
        config = OTConfig(alpha=10.0, n_iter=50, tau=0.1)
        sim = ot_similarity(h_i, h_j, config)
        assert sim.dim() == 0

    def test_ablation_pooling_cosine(self):
        from motion_aware_tpsgg.transport import pooling_cosine_similarity
        h_i = torch.randn(4, 16)
        h_j = torch.randn(6, 16)
        sim = pooling_cosine_similarity(h_i, h_j)
        assert sim.dim() == 0
        assert -1.01 <= sim.item() <= 1.01

    def test_ablation_pooling_l2(self):
        from motion_aware_tpsgg.transport import pooling_l2_similarity
        h_i = torch.randn(4, 16)
        h_j = torch.randn(4, 16)
        sim = pooling_l2_similarity(h_i, h_j)
        assert sim.dim() == 0

    def test_get_similarity_fn(self):
        from motion_aware_tpsgg.transport import get_similarity_fn, OTConfig
        for method in ["optimal_transport", "pooling_cosine", "pooling_l2"]:
            fn = get_similarity_fn(method, OTConfig(n_iter=10, tau=0.1))
            result = fn(torch.randn(3, 8), torch.randn(4, 8))
            assert result.dim() == 0


# ==================== Metrics Tests ====================

class TestMetrics:
    def test_viou_identical(self):
        from motion_aware_tpsgg.metrics import compute_viou
        tube = np.ones((4, 8, 8), dtype=bool)
        assert compute_viou(tube, tube) == 1.0

    def test_viou_no_overlap(self):
        from motion_aware_tpsgg.metrics import compute_viou
        tube_a = np.zeros((4, 8, 8), dtype=bool)
        tube_b = np.zeros((4, 8, 8), dtype=bool)
        tube_a[:, :4, :] = True
        tube_b[:, 4:, :] = True
        assert compute_viou(tube_a, tube_b) == 0.0

    def test_viou_partial(self):
        from motion_aware_tpsgg.metrics import compute_viou
        tube_a = np.zeros((4, 8, 8), dtype=bool)
        tube_b = np.zeros((4, 8, 8), dtype=bool)
        tube_a[:, :6, :] = True
        tube_b[:, 2:, :] = True
        viou = compute_viou(tube_a, tube_b)
        assert 0.0 < viou < 1.0

    def test_viou_with_torch(self):
        from motion_aware_tpsgg.metrics import compute_viou
        tube_a = torch.ones(4, 8, 8)
        tube_b = torch.ones(4, 8, 8)
        assert compute_viou(tube_a, tube_b) == 1.0

    def test_recall_at_k_perfect(self):
        from motion_aware_tpsgg.metrics import recall_at_k, PredictedTriplet, GroundTruthTriplet
        gt = [GroundTruthTriplet(subject_cat=0, object_cat=1, relation_cat=2)]
        pred = [PredictedTriplet(subject_cat=0, object_cat=1, relation_cat=2, score=1.0)]
        r = recall_at_k(pred, gt, k=20, viou_threshold=0.0)
        assert r == 1.0

    def test_recall_at_k_miss(self):
        from motion_aware_tpsgg.metrics import recall_at_k, PredictedTriplet, GroundTruthTriplet
        gt = [GroundTruthTriplet(subject_cat=0, object_cat=1, relation_cat=2)]
        pred = [PredictedTriplet(subject_cat=0, object_cat=1, relation_cat=3, score=1.0)]
        r = recall_at_k(pred, gt, k=20, viou_threshold=0.0)
        assert r == 0.0

    def test_mean_recall_at_k(self):
        from motion_aware_tpsgg.metrics import mean_recall_at_k, PredictedTriplet, GroundTruthTriplet
        gt1 = [GroundTruthTriplet(subject_cat=0, object_cat=1, relation_cat=0)]
        gt2 = [GroundTruthTriplet(subject_cat=0, object_cat=1, relation_cat=1)]
        pred1 = [PredictedTriplet(subject_cat=0, object_cat=1, relation_cat=0, score=1.0)]
        pred2 = [PredictedTriplet(subject_cat=0, object_cat=1, relation_cat=1, score=1.0)]
        mr = mean_recall_at_k([pred1, pred2], [gt1, gt2], k=20, viou_threshold=0.0)
        assert mr == 1.0

    def test_recall_empty_predictions(self):
        from motion_aware_tpsgg.metrics import recall_at_k, GroundTruthTriplet
        gt = [GroundTruthTriplet(subject_cat=0, object_cat=1, relation_cat=2)]
        r = recall_at_k([], gt, k=20, viou_threshold=0.0)
        assert r == 0.0


# ==================== Sampling Tests ====================

class TestSampling:
    def test_shuffle_tube(self):
        from motion_aware_tpsgg.sampling import shuffle_tube
        tube = torch.arange(24).reshape(4, 6).float()
        shuffled = shuffle_tube(tube)
        assert shuffled.shape == tube.shape
        # Content should be same rows, different order (with high probability)
        assert set(tuple(r.tolist()) for r in tube) == set(tuple(r.tolist()) for r in shuffled)

    def test_positive_sampler(self):
        from motion_aware_tpsgg.sampling import Triplet, PositiveSampler
        t1 = Triplet("vid1", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8))
        t2 = Triplet("vid2", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8))
        t3 = Triplet("vid1", 3, 4, 5, torch.randn(4, 8), torch.randn(4, 8))

        sampler = PositiveSampler([t1, t2, t3])
        pos = sampler.sample(t1)
        assert pos is not None
        assert pos.video_id != t1.video_id
        assert pos.category_key == t1.category_key

    def test_positive_sampler_no_match(self):
        from motion_aware_tpsgg.sampling import Triplet, PositiveSampler
        t1 = Triplet("vid1", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8))
        t2 = Triplet("vid1", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8))
        sampler = PositiveSampler([t1, t2])
        # Both from same video, no positive possible
        pos = sampler.sample(t1)
        assert pos is None

    def test_shuffle_negative_sampler(self):
        from motion_aware_tpsgg.sampling import Triplet, ShuffleNegativeSampler
        t = Triplet("vid1", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8),
                     motion_strength=10.0)
        t.build_anchor()
        sampler = ShuffleNegativeSampler(gamma=9.0, num_negatives=2)
        negs = sampler.sample(t)
        assert len(negs) == 2
        assert negs[0].shape == t.anchor_repr.shape

    def test_shuffle_sampler_weak_motion(self):
        from motion_aware_tpsgg.sampling import Triplet, ShuffleNegativeSampler
        t = Triplet("vid1", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8),
                     motion_strength=5.0)
        t.build_anchor()
        sampler = ShuffleNegativeSampler(gamma=9.0)
        negs = sampler.sample(t)
        assert len(negs) == 0

    def test_triplet_negative_sampler(self):
        from motion_aware_tpsgg.sampling import Triplet, TripletNegativeSampler
        anchor = Triplet("vid1", 0, 1, 2, torch.randn(4, 8), torch.randn(4, 8))
        anchor.build_anchor()
        others = [
            Triplet("vid1", 0, 1, 3, torch.randn(4, 8), torch.randn(4, 8)),  # shares sub+rel
            Triplet("vid1", 5, 6, 7, torch.randn(4, 8), torch.randn(4, 8)),  # shares nothing
        ]
        for t in others:
            t.build_anchor()

        sampler = TripletNegativeSampler(num_negatives=2)
        negs = sampler.sample(anchor, others)
        assert len(negs) == 2


# ==================== Model Tests ====================

class TestModels:
    def test_transformer_encoder(self):
        from motion_aware_tpsgg.models import TransformerTubeEncoder, ModelConfig
        cfg = ModelConfig(input_dim=32, hidden_dim=32, num_heads=2, max_tube_length=16)
        enc = TransformerTubeEncoder(cfg)
        x = torch.randn(2, 8, 32)
        out = enc(x)
        assert out.shape == (2, 8, 32)

    def test_conv_encoder(self):
        from motion_aware_tpsgg.models import ConvolutionTubeEncoder, ModelConfig
        cfg = ModelConfig(input_dim=32, hidden_dim=32)
        enc = ConvolutionTubeEncoder(cfg)
        x = torch.randn(2, 8, 32)
        out = enc(x)
        assert out.shape == (2, 8, 32)

    def test_relation_classifier(self):
        from motion_aware_tpsgg.models import RelationClassifier, ModelConfig
        cfg = ModelConfig(hidden_dim=32, num_relation_classes=10)
        cls = RelationClassifier(cfg)
        h_sub = torch.randn(4, 32)
        h_obj = torch.randn(4, 32)
        logits = cls(h_sub, h_obj)
        assert logits.shape == (4, 10)

    def test_full_model_forward(self):
        from motion_aware_tpsgg.models import TPSGGModel, ModelConfig
        cfg = ModelConfig(input_dim=32, hidden_dim=32, num_relation_classes=10,
                          tube_encoder_type="convolution")
        model = TPSGGModel(cfg)
        sub = torch.randn(2, 8, 32)
        obj = torch.randn(2, 8, 32)
        out = model(sub, obj)
        assert out["logits"].shape == (2, 10)
        assert out["h_sub"].shape == (2, 32)
        assert out["H_sub"].shape == (2, 8, 32)


# ==================== Loss Tests ====================

class TestLosses:
    def test_info_nce(self):
        from motion_aware_tpsgg.losses import info_nce_loss
        sim_pos = torch.tensor(5.0)
        sim_neg = torch.tensor([1.0, 2.0, 3.0])
        loss = info_nce_loss(sim_pos, sim_neg)
        assert loss.item() >= 0.0

    def test_relation_ce(self):
        from motion_aware_tpsgg.losses import RelationCrossEntropyLoss
        ce = RelationCrossEntropyLoss(num_classes=10)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = ce(logits, targets)
        assert loss.item() >= 0.0

    def test_combined_loss(self):
        from motion_aware_tpsgg.losses import MotionAwareContrastiveLoss, LossConfig
        cfg = LossConfig(similarity_method="pooling_cosine")
        loss_fn = MotionAwareContrastiveLoss(cfg, num_relation_classes=10)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets)
        assert "total" in result
        assert result["total"].item() >= 0.0


# ==================== Dataset Tests ====================

class TestSyntheticDataset:
    def test_creation(self):
        from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
        ds = SyntheticDebugDataset(num_samples=5, seed=42)
        assert len(ds) == 5

    def test_sample_structure(self):
        from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
        ds = SyntheticDebugDataset(num_samples=3, feature_dim=32, num_frames=4)
        sample = ds[0]
        assert sample.tube_features.shape[1] == 4
        assert sample.tube_features.shape[2] == 32
        assert len(sample.relations) > 0

    def test_with_masks(self):
        from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
        ds = SyntheticDebugDataset(num_samples=2, include_masks=True)
        sample = ds[0]
        assert sample.mask_tubes is not None

    def test_collate(self):
        from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
        from motion_aware_tpsgg.datasets.base import collate_relation_pairs
        ds = SyntheticDebugDataset(num_samples=3, feature_dim=32, num_frames=4)
        samples = [ds[i] for i in range(3)]
        batch = collate_relation_pairs(samples, pad_length=8)
        assert batch.subject_tubes.dim() == 3
        assert batch.relation_labels.dim() == 1


# ==================== Training Step Test ====================

class TestTrainingStep:
    def test_one_step(self):
        """Test that one training step runs without error."""
        from motion_aware_tpsgg.models import TPSGGModel, ModelConfig
        from motion_aware_tpsgg.losses import MotionAwareContrastiveLoss, LossConfig
        from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
        from motion_aware_tpsgg.datasets.base import collate_relation_pairs

        model_cfg = ModelConfig(input_dim=32, hidden_dim=32, num_relation_classes=5,
                                tube_encoder_type="convolution", num_encoder_layers=1)
        model = TPSGGModel(model_cfg)

        loss_cfg = LossConfig(similarity_method="pooling_cosine")
        loss_fn = MotionAwareContrastiveLoss(loss_cfg, num_relation_classes=5)

        ds = SyntheticDebugDataset(num_samples=4, feature_dim=32, num_frames=4,
                                    num_relation_classes=5)
        samples = [ds[i] for i in range(4)]
        batch = collate_relation_pairs(samples, pad_length=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Forward
        outputs = model(batch.subject_tubes, batch.object_tubes)
        losses = loss_fn(outputs["logits"], batch.relation_labels)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        assert losses["total"].item() >= 0.0
