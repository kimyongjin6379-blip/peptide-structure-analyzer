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
        각 잔기 위치의 중요도 분석 (pseudo-perplexity 기반, 배치 최적화)

        모든 마스킹 위치를 한 번의 배치 forward pass로 처리하여
        기존 대비 ~서열길이 배 빠름.

        높은 값 = 모델이 "예상치 못한" 잔기 → 기능적으로 중요할 가능성

        Returns:
            dict: {position: importance_score}
        """
        self.load_model()

        import torch

        seq_len = len(sequence)
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # 모든 마스킹 위치를 한 번에 배치로 구성
        # batch_tokens shape: (1, seq_len+2) — [CLS] + seq + [EOS]
        masked_batch = batch_tokens.repeat(seq_len, 1)  # (seq_len, seq_len+2)
        for i in range(seq_len):
            masked_batch[i, i + 1] = self.alphabet.mask_idx

        importance = {}

        with torch.no_grad():
            # 메모리 제한을 고려한 미니배치 처리
            mini_batch_size = 32
            all_probs = []

            for start in range(0, seq_len, mini_batch_size):
                end = min(start + mini_batch_size, seq_len)
                mini_batch = masked_batch[start:end]
                logits = self.model(mini_batch)["logits"]

                for local_idx in range(end - start):
                    global_idx = start + local_idx
                    probs = torch.softmax(logits[local_idx, global_idx + 1], dim=-1)
                    aa_idx = self.alphabet.get_idx(sequence[global_idx])
                    prob = probs[aa_idx].item()
                    importance[global_idx + 1] = round(-np.log(prob + 1e-10), 4)

        return importance

    def get_fitness_score(self, sequence: str) -> float:
        """
        서열의 전체 fitness score 계산 (pseudo-log-likelihood)
        낮을수록 진화적으로 타당한 서열

        Returns normalized score between 0 and 1 (1 = best)
        """
        importance = self.get_residue_importance(sequence)
        avg_nll = np.mean(list(importance.values()))
        # Convert to 0-1 score (lower NLL = higher fitness)
        # Typical range: 1-8 for NLL, so sigmoid-like normalization
        score = 1.0 / (1.0 + np.exp((avg_nll - 4.0) / 1.5))
        return round(float(score), 4)

    def get_batch_fitness_scores(self, sequences: list, batch_size: int = 16) -> list:
        """
        임베딩 기반 빠른 fitness scoring (배치 처리)

        마스킹 방식 대비 ~10배 빠름. 서열당 forward pass 1회.
        임베딩 벡터의 노름과 내부 분산을 활용하여 fitness를 추정.

        원리: 자연 단백질은 ESM-2 임베딩 공간에서 특정 분포를 형성함.
        - 임베딩 노름이 적정 범위 → 자연스러운 서열
        - 잔기 간 임베딩 분산이 적정 → 일관된 구조적 맥락

        Args:
            sequences: 서열 문자열 리스트
            batch_size: 배치 크기

        Returns:
            fitness score 리스트 (0~1)
        """
        self.load_model()
        import torch

        all_scores = []
        num_layers = self._model_info.get("layers", 6)

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            data = [(f"p_{j}", seq) for j, seq in enumerate(batch_seqs)]

            try:
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)

                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[num_layers])
                    logits = results["logits"]
                    representations = results["representations"][num_layers]

                for j, seq in enumerate(batch_seqs):
                    seq_len = len(seq)

                    # Method 1: Embedding norm-based fitness
                    emb = representations[j, 1:seq_len + 1, :]  # (L, D)
                    mean_emb = emb.mean(dim=0)
                    norm = torch.norm(mean_emb).item()
                    # Natural proteins typically have norm in range 2-8
                    norm_score = 1.0 / (1.0 + np.exp(-(norm - 3.0) / 1.5))

                    # Method 2: Log-likelihood from logits (no masking needed)
                    # Use the model's output logits directly as an approximation
                    token_logits = logits[j, 1:seq_len + 1, :]  # (L, vocab)
                    token_probs = torch.softmax(token_logits, dim=-1)

                    total_log_prob = 0.0
                    for pos in range(seq_len):
                        aa_idx = self.alphabet.get_idx(seq[pos])
                        prob = token_probs[pos, aa_idx].item()
                        total_log_prob += np.log(prob + 1e-10)

                    avg_log_prob = total_log_prob / seq_len
                    # Typical range: -8 to -1, higher is better
                    logit_score = 1.0 / (1.0 + np.exp(-(avg_log_prob + 3.0) / 1.5))

                    # Combined: logit-based (more informative) + norm-based
                    fitness = logit_score * 0.7 + norm_score * 0.3
                    all_scores.append(round(float(fitness), 4))

            except Exception as e:
                # Fallback: assign neutral scores for failed batches
                all_scores.extend([0.5] * len(batch_seqs))

        return all_scores

    def get_improvement_suggestions(self, sequence: str, top_n: int = 3) -> list:
        """
        서열의 각 위치에서 더 좋은 아미노산을 제안

        Returns list of dicts with position, current_aa, suggested_aa, probability_gain
        """
        self.load_model()
        import torch

        suggestions = []
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        for i in range(len(sequence)):
            masked = batch_tokens.clone()
            masked[0, i + 1] = self.alphabet.mask_idx

            with torch.no_grad():
                logits = self.model(masked)["logits"]

            probs = torch.softmax(logits[0, i + 1], dim=-1)
            current_aa = sequence[i]
            current_idx = self.alphabet.get_idx(current_aa)
            current_prob = probs[current_idx].item()

            # Find the best amino acid for this position
            best_prob, best_idx = probs.max(dim=-1)
            best_aa = self.alphabet.get_tok(best_idx.item())

            if best_aa != current_aa and best_aa in "ACDEFGHIKLMNPQRSTVWY":
                gain = best_prob.item() - current_prob
                if gain > 0.05:  # Only suggest if meaningful improvement
                    suggestions.append({
                        "position": i + 1,
                        "current_aa": current_aa,
                        "suggested_aa": best_aa,
                        "current_prob": round(current_prob, 4),
                        "suggested_prob": round(best_prob.item(), 4),
                        "probability_gain": round(gain, 4),
                        "mutation": f"{current_aa}{i+1}{best_aa}"
                    })

        # Sort by gain, return top_n
        suggestions.sort(key=lambda x: x["probability_gain"], reverse=True)
        return suggestions[:top_n]

    def get_motif_embedding(self, motif: str) -> np.ndarray:
        """짧은 모티프의 임베딩 (유사 모티프 검색용)"""
        return self.get_sequence_embedding(motif)

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
