"""
PLM Embedder - ESM-2 Protein Language Model 임베딩 추출 모듈

Railway 환경에서 ESM-2 모델을 로컬 로딩하여:
- 서열 임베딩 (representation) 추출
- Zero-shot fitness prediction (변이 효과 예측)
- 임베딩 기반 서열 유사도 계산
"""

import numpy as np
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class PLMEmbedder:
    """ESM-2 모델 기반 단백질 서열 임베딩 추출기"""

    # 사용 가능한 ESM-2 모델 목록 (작은 것부터)
    AVAILABLE_MODELS = {
        "esm2_t6_8M": {"layers": 6, "dim": 320, "params": "8M"},
        "esm2_t12_35M": {"layers": 12, "dim": 480, "params": "35M"},
        "esm2_t30_150M": {"layers": 30, "dim": 640, "params": "150M"},
        "esm2_t33_650M": {"layers": 33, "dim": 1280, "params": "650M"},
    }

    def __init__(self, model_name: str = "esm2_t6_8M", device: str = "cpu"):
        """
        Args:
            model_name: 사용할 ESM-2 모델
                - "esm2_t6_8M": 가볍고 빠름 (Railway 기본 추천)
                - "esm2_t12_35M": 중간 성능
                - "esm2_t30_150M": 높은 성능
                - "esm2_t33_650M": 최고 성능 (RAM 4GB+ 필요)
            device: "cpu" (Railway는 CPU 전용)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._model_info = self.AVAILABLE_MODELS.get(model_name, {})

        # 캐시 디렉토리
        self.cache_dir = Path(__file__).parent.parent / "data" / "cache" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """ESM-2 모델 로딩 (첫 호출 시 자동 다운로드)"""
        if self.model is not None:
            return  # 이미 로딩됨

        try:
            import torch
            import esm

            logger.info(f"Loading ESM-2 model: {self.model_name}...")

            # 모델별 로딩 함수 매핑
            model_loaders = {
                "esm2_t6_8M": esm.pretrained.esm2_t6_8M_UR50D,
                "esm2_t12_35M": esm.pretrained.esm2_t12_35M_UR50D,
                "esm2_t30_150M": esm.pretrained.esm2_t30_150M_UR50D,
                "esm2_t33_650M": esm.pretrained.esm2_t33_650M_UR50D,
            }

            loader = model_loaders.get(self.model_name)
            if loader is None:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.model, self.alphabet = loader()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded: {self._model_info.get('params', '?')} parameters, "
                       f"dim={self._model_info.get('dim', '?')}")

        except ImportError:
            raise ImportError(
                "ESM 패키지가 필요합니다. `pip install fair-esm` 으로 설치하세요."
            )

    def get_embedding(self, sequence: str, layer: int = -1) -> np.ndarray:
        """
        서열의 잔기별 임베딩 추출

        Args:
            sequence: 아미노산 서열 (예: "ACDEFGHIKLMNPQRSTVWY")
            layer: 추출할 레이어 (-1이면 마지막 레이어)

        Returns:
            np.ndarray: (L, D) 차원 임베딩
                L = 서열 길이, D = 임베딩 차원
        """
        self.load_model()

        import torch

        # 캐시 확인
        cache_key = self._cache_key(sequence, layer)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # 레이어 번호 처리
        num_layers = self._model_info.get("layers", 6)
        if layer == -1:
            layer = num_layers

        # 배치 준비
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # 추론
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer], return_contacts=False)

        # (1, L+2, D) → (L, D) : BOS/EOS 토큰 제거
        embedding = results["representations"][layer][0, 1:-1, :].cpu().numpy()

        # 캐시 저장
        self._save_cache(cache_key, embedding)

        return embedding

    def get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """
        서열 전체의 평균 풀링 임베딩

        Returns:
            np.ndarray: (D,) 차원 벡터
        """
        residue_embeddings = self.get_embedding(sequence)
        return residue_embeddings.mean(axis=0)

    def get_batch_embeddings(self, sequences: list, batch_size: int = 8) -> list:
        """
        여러 서열의 임베딩을 배치로 추출

        Args:
            sequences: 서열 리스트
            batch_size: 배치 크기

        Returns:
            list of np.ndarray: 서열별 (D,) 임베딩 리스트
        """
        self.load_model()

        import torch

        all_embeddings = []
        num_layers = self._model_info.get("layers", 6)

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            data = [(f"protein_{j}", seq) for j, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[num_layers])

            representations = results["representations"][num_layers]

            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                emb = representations[j, 1:seq_len + 1, :].cpu().numpy()
                all_embeddings.append(emb.mean(axis=0))

        return all_embeddings

    def zero_shot_score(self, sequence: str, mutations: list) -> dict:
        """
        Zero-shot 변이 효과 예측 (masked marginal scoring)

        pLM이 학습한 진화적 정보를 활용하여, 특정 변이가
        단백질 기능에 미치는 영향을 추가 학습 없이 예측.

        Args:
            sequence: 야생형(wild-type) 서열
            mutations: 변이 리스트, 예: ["A5G", "L10V", "K15R"]
                형식: {원래AA}{위치}{변이AA}

        Returns:
            dict: {
                "A5G": {"score": -0.5, "effect": "deleterious", "wt_prob": 0.8, "mt_prob": 0.3},
                ...
            }
        """
        self.load_model()

        import torch

        results = {}

        for mutation in mutations:
            try:
                wt_aa = mutation[0]
                position = int(mutation[1:-1]) - 1  # 0-based index
                mt_aa = mutation[-1]

                # 검증
                if position < 0 or position >= len(sequence):
                    results[mutation] = {"error": f"Position {position + 1} out of range"}
                    continue

                if sequence[position] != wt_aa:
                    results[mutation] = {
                        "error": f"Expected {wt_aa} at position {position + 1}, found {sequence[position]}"
                    }
                    continue

                # Masked marginal probability 계산
                data = [("protein", sequence)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)

                # 해당 위치를 마스크
                masked_tokens = batch_tokens.clone()
                masked_tokens[0, position + 1] = self.alphabet.mask_idx  # +1 for BOS

                with torch.no_grad():
                    logits = self.model(masked_tokens)["logits"]

                # 해당 위치의 확률 분포
                probs = torch.softmax(logits[0, position + 1], dim=-1)

                wt_idx = self.alphabet.get_idx(wt_aa)
                mt_idx = self.alphabet.get_idx(mt_aa)

                wt_prob = probs[wt_idx].item()
                mt_prob = probs[mt_idx].item()

                # Log-likelihood ratio
                score = np.log(mt_prob / (wt_prob + 1e-10))

                # 효과 판정
                if score > 0.5:
                    effect = "beneficial"
                elif score > -0.5:
                    effect = "neutral"
                else:
                    effect = "deleterious"

                results[mutation] = {
                    "score": round(score, 4),
                    "effect": effect,
                    "wt_prob": round(wt_prob, 4),
                    "mt_prob": round(mt_prob, 4),
                    "wt_aa": wt_aa,
                    "mt_aa": mt_aa,
                    "position": position + 1
                }

            except Exception as e:
                results[mutation] = {"error": str(e)}

        return results

    def compute_similarity(self, seq1: str, seq2: str) -> float:
        """
        임베딩 기반 서열 유사도 계산 (코사인 유사도)

        기존 peptide_analyzer.py의 조성 기반 유사도보다
        훨씬 정확한 기능적 유사도를 제공.

        Returns:
            float: 0~1 사이 유사도
        """
        emb1 = self.get_sequence_embedding(seq1)
        emb2 = self.get_sequence_embedding(seq2)

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        similarity = dot_product / (norm1 * norm2 + 1e-10)
        return float(np.clip(similarity, 0, 1))

    def compute_similarity_matrix(self, sequences: list) -> np.ndarray:
        """
        여러 서열 간 유사도 행렬 계산

        Returns:
            np.ndarray: (N, N) 유사도 행렬
        """
        embeddings = self.get_batch_embeddings(sequences)
        n = len(embeddings)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                dot = np.dot(embeddings[i], embeddings[j])
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                sim = dot / (norm_i * norm_j + 1e-10)
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix

    def get_residue_importance(self, sequence: str) -> dict:
        """
        각 잔기 위치의 중요도 분석 (pseudo-perplexity 기반)

        높은 값 = 모델이 "예상치 못한" 잔기 → 기능적으로 중요할 가능성

        Returns:
            dict: {position: importance_score}
        """
        self.load_model()

        import torch

        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        importance = {}

        with torch.no_grad():
            for i in range(len(sequence)):
                masked = batch_tokens.clone()
                masked[0, i + 1] = self.alphabet.mask_idx

                logits = self.model(masked)["logits"]
                probs = torch.softmax(logits[0, i + 1], dim=-1)

                aa_idx = self.alphabet.get_idx(sequence[i])
                prob = probs[aa_idx].item()

                # -log(p)가 높을수록 "놀라운" 잔기 = 중요
                importance[i + 1] = round(-np.log(prob + 1e-10), 4)

        return importance

    def get_model_info(self) -> dict:
        """현재 로드된 모델 정보"""
        return {
            "model_name": self.model_name,
            "parameters": self._model_info.get("params", "unknown"),
            "embedding_dim": self._model_info.get("dim", "unknown"),
            "num_layers": self._model_info.get("layers", "unknown"),
            "device": self.device,
            "loaded": self.model is not None
        }

    # ---- 캐시 관리 ----

    def _cache_key(self, sequence: str, layer: int) -> str:
        content = f"{self.model_name}_{sequence}_{layer}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            try:
                return np.load(str(cache_path))
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, embedding: np.ndarray):
        cache_path = self.cache_dir / f"{key}.npy"
        try:
            np.save(str(cache_path), embedding)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
