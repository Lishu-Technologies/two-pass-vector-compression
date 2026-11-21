"""Build 64-byte product codes and scan them by cosine similarity."""

import numpy as np
import torch


def _unit_rows(values):
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 512 or not np.isfinite(rows).all():
        raise ValueError("Expected a finite matrix with 512 columns")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms < 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Every vector must have a finite, nonzero length")
    return np.ascontiguousarray(rows / norms)


def _check_codes(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 64 or codes.dtype != np.uint8:
        raise ValueError("Expected an N by 64 matrix with dtype uint8")
    return codes


class VectorCodec:
    """One shared float32 codebook, with 256 eight-dimensional centers per block."""

    def __init__(self, centers):
        centers = np.array(centers, dtype=np.float32, order="C", copy=True)
        if centers.shape != (64, 256, 8) or not np.isfinite(centers).all():
            raise ValueError("Expected finite centers of shape (64, 256, 8)")
        self.centers = centers

    @torch.inference_mode()
    def encode(self, gallery, probes, *, passes=2, initial_codes=None, device="cpu"):
        """Refine at most one symbol per row per pass. All inputs share one basis."""
        if passes not in (0, 1, 2):
            raise ValueError("passes must be 0, 1, or 2")
        gallery = _unit_rows(gallery)
        if initial_codes is not None:
            initial_codes = _check_codes(initial_codes)
            if len(initial_codes) != len(gallery):
                raise ValueError("Expected one initial code per gallery row")
        if passes:
            bank = torch.as_tensor(_unit_rows(probes), device=device)
            if len(bank) < 32:
                raise ValueError("Provide at least 32 calibration queries")

        centers = torch.as_tensor(self.centers, device=device)
        square = centers.square().sum(-1)
        blocks = torch.arange(64, device=device)
        output = np.empty((len(gallery), 64), dtype=np.uint8)
        for start in range(0, len(gallery), 128):
            rows = slice(start, start + 128)
            target = torch.as_tensor(gallery[rows], device=device)
            parts = target.reshape(-1, 64, 8)
            distances = square - 2 * torch.einsum("bmd,mkd->bmk", parts, centers)
            if initial_codes is None:
                current = distances.argmin(2)
            else:
                current = torch.as_tensor(initial_codes[rows], device=device).long()
            if passes:
                probe_ids = (target @ bank.T).argsort(dim=1, descending=True, stable=True)[:, :32]
                local = bank[probe_ids]
                exact = (local * target[:, None]).sum(-1)
                for _ in range(passes):
                    current = self._update(parts, local, exact, centers, blocks, distances, current)
            output[rows] = current.to(torch.uint8).cpu().numpy()
        return output

    @staticmethod
    def _update(parts, local, exact, centers, blocks, distances, current):
        # Try one alternative per block. Each option changes only that block.
        alternative = distances.scatter(2, current.unsqueeze(-1), torch.inf).argmin(2)
        original = centers[blocks, current]
        proposed = centers[blocks, alternative]
        shift = proposed - original
        decoded = original.flatten(1)
        norm_sq = decoded.square().sum(1)
        norm = norm_sq.sqrt().clamp_min(1e-10)
        new_sq = norm_sq[:, None] + (proposed.square() - original.square()).sum(2)
        new_norm = new_sq.clamp_min(1e-20).sqrt()

        dots = (local * decoded[:, None]).sum(-1)
        delta = torch.einsum("bkmd,bmd->bkm", local.reshape(-1, 32, 64, 8), shift)
        scores = (dots[:, :, None] + delta) / new_norm[:, None]
        alignment = (parts * original).sum((1, 2))
        new_alignment = (alignment[:, None] + (parts * shift).sum(2)) / new_norm
        penalty = 0.1 / 512
        cost = (dots / norm[:, None] - exact).square().mean(1)
        cost += penalty * (2 - 2 * alignment / norm)
        new_cost = (scores - exact[:, :, None]).square().mean(1)
        new_cost += penalty * (2 - 2 * new_alignment)
        best, block = new_cost.min(1)
        rows = torch.where(best < cost - 1e-12)[0]
        current[rows, block[rows]] = alternative[rows, block[rows]]
        return current

    def scores(self, query, codes):
        """Cosine scores in gallery order, without expanding the gallery to floats."""
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (512,):
            raise ValueError("Expected one query vector of shape (512,)")
        query = _unit_rows(query[None]).reshape(64, 8)
        codes = _check_codes(codes)
        table = np.einsum("md,mkd->mk", query, self.centers)
        square = np.einsum("mkd,mkd->mk", self.centers, self.centers)
        dots = np.zeros(len(codes), dtype=np.float32)
        norm_sq = np.zeros_like(dots)
        for block, symbols in enumerate(codes.T):
            dots += table[block, symbols]
            norm_sq += square[block, symbols]
        return dots / np.sqrt(np.maximum(norm_sq, 1e-20))
