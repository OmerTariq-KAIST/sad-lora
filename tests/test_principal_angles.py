"""Unit tests for Grassmannian geometry utilities."""

import pytest
import torch
import math

from src.utils.grassmannian import (
    principal_angles,
    alignment_score,
    subspace_distance,
    spectral_overlap_matrix,
)


class TestPrincipalAngles:
    def test_identical_subspaces(self):
        """All angles should be 0 for identical subspaces."""
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, r))
        angles = principal_angles(Q, Q)
        assert angles.max().item() < 1e-5

    def test_orthogonal_subspaces(self):
        """All angles should be pi/2 for orthogonal subspaces."""
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, 2 * r))
        A, B = Q[:, :r], Q[:, r:]
        angles = principal_angles(A, B)
        assert (angles - math.pi / 2).abs().max().item() < 1e-4

    def test_known_angle(self):
        """Test with a known 45-degree rotation in 2D subspace."""
        d, r = 4, 2
        A = torch.eye(d, r)
        theta = math.pi / 4
        rot = torch.eye(d)
        rot[0, 0] = rot[1, 1] = math.cos(theta)
        rot[0, 1] = -math.sin(theta)
        rot[1, 0] = math.sin(theta)
        B = rot @ A
        angles = principal_angles(A, B)
        # One angle should be theta, the other 0 (rotation only affects one plane)
        assert angles.min().item() < 1e-4  # One direction unchanged
        assert abs(angles.max().item() - theta) < 1e-4

    def test_cosine_mode(self):
        """return_cos should give cosines in descending order."""
        d, r = 64, 4
        A = torch.randn(d, r)
        B = torch.randn(d, r)
        cos_vals = principal_angles(A, B, return_cos=True)
        # Cosines should be in [0, 1] and descending
        assert (cos_vals >= -1e-6).all() and (cos_vals <= 1 + 1e-6).all()
        diffs = cos_vals[:-1] - cos_vals[1:]
        assert (diffs >= -1e-6).all()  # descending


class TestAlignmentScore:
    def test_perfect_alignment(self):
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, r))
        score = alignment_score(Q, Q)
        assert abs(score - 1.0) < 1e-5

    def test_zero_alignment(self):
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, 2 * r))
        score = alignment_score(Q[:, :r], Q[:, r:])
        assert abs(score) < 1e-5

    def test_bounded(self):
        for _ in range(10):
            d, r = 32, 3
            score = alignment_score(torch.randn(d, r), torch.randn(d, r))
            assert -1e-6 <= score <= 1.0 + 1e-6


class TestSubspaceDistance:
    def test_zero_distance_identical(self):
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, r))
        for metric in ("projection", "geodesic", "chordal"):
            dist = subspace_distance(Q, Q, metric=metric)
            assert dist < 1e-5

    def test_positive_distance(self):
        d, r = 64, 4
        A = torch.randn(d, r)
        B = torch.randn(d, r)
        for metric in ("projection", "geodesic", "chordal"):
            dist = subspace_distance(A, B, metric=metric)
            assert dist > 0


class TestSpectralOverlap:
    def test_identity_overlap(self):
        """Overlap with itself should be identity-like."""
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, r))
        overlap = spectral_overlap_matrix(Q, Q)
        # Should be close to identity (permutation)
        assert overlap.shape == (r, r)
        assert overlap.max().item() > 0.99

    def test_orthogonal_overlap(self):
        """Overlap between orthogonal subspaces should be ~0."""
        d, r = 64, 4
        Q, _ = torch.linalg.qr(torch.randn(d, 2 * r))
        overlap = spectral_overlap_matrix(Q[:, :r], Q[:, r:])
        assert overlap.max().item() < 1e-4
