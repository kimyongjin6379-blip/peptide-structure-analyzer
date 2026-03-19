"""
PLM Fine-tuner - ESM-2를 펩톤 유래 펩타이드 데이터에 특화시키는 모듈

Masked Language Modeling (MLM) 방식으로 Fine-tuning:
- 라벨 데이터 불필요 (서열만 있으면 됨)
- ESM-2의 확률 분포를 펩톤 서열 쪽으로 이동
- 결과: 더 정확한 임베딩, 더 현실적인 변이 예측, 더 나은 서열 생성

구 버전 (범용 ESM-2) vs 신 버전 (펩톤 특화 ESM-2) 비교 기능 포함
"""

import numpy as np
import json
import logging
import time
import copy
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


class PeptoneTrainingDataBuilder:
    """
    Fine-tuning용 학습 데이터 수집/구축

    데이터 소스:
    1. bioactive_peptide_db.json의 알려진 모티프 서열
    2. Markov 생성 서열 (각 샘플별)
    3. 알려진 식품 유래 생리활성 펩타이드 DB
    4. 사용자 직접 입력 서열
    """

    # 알려진 식품 유래 생리활성 펩타이드 (추가 학습 데이터)
    KNOWN_FOOD_PEPTIDES = [
        # ACE inhibitory (항고혈압)
        "IPP", "VPP", "LKP", "IKP", "LRW", "VRF", "LRF", "GKP", "IQP",
        "LKPNM", "FFVAPFPE", "KVLPVPE", "AVPYPQR", "FALPQY", "TTMPLW",
        "RYLGY", "AYFYPEL", "YQEPVLGP", "RPKHPIKHQ", "TQVYSR",
        # Antioxidant (항산화)
        "YFCLT", "YQEPVLGP", "PHFL", "YWCL", "FCLLT", "WYSL",
        "LKPTPEGDL", "VLPVPQK", "PYPQ", "EL", "AL", "VKEAMAPK",
        "HVASF", "HGSLH", "FHGSH", "FGHPY", "AWWGHL",
        # Antimicrobial (항균)
        "RRWQWR", "RWQWRWQR", "RRGWALRL", "FLGALWNVR", "RLWRIVVIRK",
        "KWCFRVCYRGICYRRCR", "GLFGAIAGFI", "GIGKFLHSAGKF",
        # DPP-IV inhibitory (항당뇨)
        "IPI", "WR", "LPQNIPPL", "IPIQY", "IPA", "LPYPY", "FLQP",
        "IPAVF", "VPITPT", "TPVVVPPF", "IPPM", "LPLPL",
        # Anti-inflammatory (항염)
        "VPP", "IPP", "PAY", "VPY", "YVPGP", "KVPQVST",
        # 발효 유래 펩타이드
        "VHVV", "KAVLG", "FDKLPGFG", "PYPQ", "NIPPLTQTPV",
        # 콜라겐 유래
        "GPP", "GAP", "GPA", "GPPGPPGPP", "GAPGAPGAP",
        # 대두 유래
        "NWGPLV", "YVVNPDNDEN", "YVVNPDNNEN", "PGTAVFK",
        # 유청 유래
        "ALPMHIR", "GLDIQK", "IPAVFK", "VAGTWY", "AASDISLLDAQSAPLR",
        # 어류 유래
        "VKAGFAWTANQQLS", "LKQELEDLLEKQE", "IVGGFPHYL",
        # 쌀 유래
        "GYGP", "GGGY", "GYYG", "GYRP", "GYGG",
        # 감마-글루타밀 펩타이드 (감칠맛/코쿠미)
        "EE", "EG", "EV", "EA", "EL", "EI", "EF", "EW", "EY",
        "EVG", "EAG", "ELG", "EIG", "ECG",
    ]

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else (
            Path(__file__).parent.parent / "data"
        )

    def build_training_set(
        self,
        include_db_motifs: bool = True,
        include_known_peptides: bool = True,
        include_markov_seqs: bool = True,
        custom_sequences: List[str] = None,
        markov_loader=None,
        markov_n_per_sample: int = 100,
        min_length: int = 3,
        max_length: int = 50,
    ) -> Dict:
        """
        학습 데이터셋 구축

        Returns:
            dict: {
                "sequences": [...],
                "sources": {...},
                "stats": {...}
            }
        """
        all_sequences = []
        sources = {
            "db_motifs": [],
            "known_peptides": [],
            "markov_generated": [],
            "custom": []
        }

        # 1. DB 모티프
        if include_db_motifs:
            db_path = self.data_dir / "bioactive_peptide_db.json"
            if db_path.exists():
                with open(db_path, 'r', encoding='utf-8') as f:
                    db = json.load(f)
                motif_seqs = [m["sequence"] for m in db.get("motifs", [])]
                sources["db_motifs"] = motif_seqs
                all_sequences.extend(motif_seqs)

        # 2. 알려진 식품 유래 펩타이드
        if include_known_peptides:
            valid = [s for s in self.KNOWN_FOOD_PEPTIDES
                     if min_length <= len(s) <= max_length]
            sources["known_peptides"] = valid
            all_sequences.extend(valid)

        # 3. Markov 생성 서열
        if include_markov_seqs and markov_loader is not None:
            try:
                from sequence_predictor import AbundancePredictor
                predictor = AbundancePredictor(markov_loader)
                samples = markov_loader.get_sample_list()

                for sample in samples:
                    try:
                        result = predictor.predict_for_sample(
                            sample,
                            length_range=(min_length, min(max_length, 15)),
                            n_sequences=markov_n_per_sample,
                            method="markov"
                        )
                        seqs = [s[0] for s in result.get("top_sequences", [])]
                        sources["markov_generated"].extend(seqs)
                        all_sequences.extend(seqs)
                    except Exception as e:
                        logger.warning(f"Markov generation failed for {sample}: {e}")
            except Exception as e:
                logger.warning(f"Markov generation skipped: {e}")

        # 4. 사용자 입력 서열
        if custom_sequences:
            valid_custom = []
            for seq in custom_sequences:
                clean = "".join(c for c in seq.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
                if min_length <= len(clean) <= max_length:
                    valid_custom.append(clean)
            sources["custom"] = valid_custom
            all_sequences.extend(valid_custom)

        # 중복 제거
        unique_sequences = list(set(all_sequences))

        # 통계
        stats = {
            "total_unique": len(unique_sequences),
            "total_raw": len(all_sequences),
            "duplicates_removed": len(all_sequences) - len(unique_sequences),
            "by_source": {k: len(v) for k, v in sources.items()},
            "length_distribution": {
                "min": min(len(s) for s in unique_sequences) if unique_sequences else 0,
                "max": max(len(s) for s in unique_sequences) if unique_sequences else 0,
                "mean": np.mean([len(s) for s in unique_sequences]) if unique_sequences else 0,
            },
            "avg_length": round(np.mean([len(s) for s in unique_sequences]), 1) if unique_sequences else 0,
        }

        return {
            "sequences": unique_sequences,
            "sources": sources,
            "stats": stats
        }


class PLMFineTuner:
    """
    ESM-2 Masked Language Modeling Fine-tuner

    펩톤 유래 펩타이드 서열로 ESM-2를 추가 학습하여
    펩톤 도메인에 특화된 언어 모델을 만듦.
    """

    def __init__(self, model_name: str = "esm2_t6_8M", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.training_history = []

        # 저장 경로
        self.save_dir = Path(__file__).parent.parent / "data" / "finetuned_models"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_base_model(self):
        """베이스 ESM-2 모델 로딩"""
        import torch
        import esm

        model_loaders = {
            "esm2_t6_8M": esm.pretrained.esm2_t6_8M_UR50D,
            "esm2_t12_35M": esm.pretrained.esm2_t12_35M_UR50D,
            "esm2_t30_150M": esm.pretrained.esm2_t30_150M_UR50D,
        }

        loader = model_loaders.get(self.model_name)
        if loader is None:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.model, self.alphabet = loader()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)

        logger.info(f"Base model loaded: {self.model_name}")

    def finetune(
        self,
        sequences: List[str],
        epochs: int = 10,
        lr: float = 1e-5,
        mask_ratio: float = 0.15,
        batch_size: int = 8,
        val_split: float = 0.1,
        freeze_layers: int = 0,
        progress_callback=None,
    ) -> Dict:
        """
        MLM Fine-tuning 실행

        Args:
            sequences: 학습용 서열 리스트
            epochs: 학습 에폭 수
            lr: 학습률 (1e-5 권장 — 너무 크면 catastrophic forgetting)
            mask_ratio: 마스킹 비율 (15% 기본, BERT와 동일)
            batch_size: 배치 크기
            val_split: 검증 데이터 비율
            freeze_layers: 하위 N개 레이어 동결 (0=전체 학습, 4=상위 2개만 학습)
            progress_callback: Streamlit 프로그레스 바 콜백

        Returns:
            dict: 학습 결과
        """
        import torch
        import torch.nn as nn

        if self.model is None:
            self.load_base_model()

        # 데이터 분할
        np.random.seed(42)
        indices = np.random.permutation(len(sequences))
        val_size = max(1, int(len(sequences) * val_split))
        val_seqs = [sequences[i] for i in indices[:val_size]]
        train_seqs = [sequences[i] for i in indices[val_size:]]

        # 레이어 동결
        if freeze_layers > 0:
            for name, param in self.model.named_parameters():
                if "layers." in name:
                    layer_num = int(name.split("layers.")[1].split(".")[0])
                    if layer_num < freeze_layers:
                        param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Frozen layers: 0~{freeze_layers-1}, Trainable: {trainable}/{total}")

        # Optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.training_history = []
        best_val_loss = float("inf")
        best_model_state = None
        start_time = time.time()

        for epoch in range(epochs):
            # ---- Training ----
            self.model.train()
            train_losses = []
            np.random.shuffle(train_seqs)

            for i in range(0, len(train_seqs), batch_size):
                batch = train_seqs[i:i + batch_size]
                loss = self._compute_mlm_loss(batch, mask_ratio)

                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

            scheduler.step()

            # ---- Validation ----
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_seqs), batch_size):
                    batch = val_seqs[i:i + batch_size]
                    loss = self._compute_mlm_loss(batch, mask_ratio)
                    if loss is not None:
                        val_losses.append(loss.item())

            avg_train = np.mean(train_losses) if train_losses else float("inf")
            avg_val = np.mean(val_losses) if val_losses else float("inf")

            # Perplexity = exp(loss)
            train_ppl = np.exp(min(avg_train, 20))  # cap to avoid overflow
            val_ppl = np.exp(min(avg_val, 20))

            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train, 4),
                "val_loss": round(avg_val, 4),
                "train_perplexity": round(train_ppl, 2),
                "val_perplexity": round(val_ppl, 2),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            self.training_history.append(epoch_info)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = copy.deepcopy(self.model.state_dict())

            if progress_callback:
                progress_callback(epoch + 1, epochs, epoch_info)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train:.4f} (PPL: {train_ppl:.1f}) | "
                f"Val Loss: {avg_val:.4f} (PPL: {val_ppl:.1f})"
            )

        # 최적 모델 복원
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.model.eval()
        total_time = time.time() - start_time

        return {
            "epochs_trained": epochs,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_perplexity": round(np.exp(min(best_val_loss, 20)), 2),
            "training_sequences": len(train_seqs),
            "validation_sequences": len(val_seqs),
            "total_time_sec": round(total_time, 1),
            "history": self.training_history,
        }

    def _compute_mlm_loss(self, sequences: List[str], mask_ratio: float):
        """MLM 손실 계산"""
        import torch

        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

        try:
            _, _, batch_tokens = self.batch_converter(data)
        except Exception:
            return None

        batch_tokens = batch_tokens.to(self.device)

        # 마스킹: 15% 랜덤 위치를 [MASK] 토큰으로 교체
        labels = batch_tokens.clone()
        mask = torch.zeros_like(batch_tokens, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            n_mask = max(1, int(seq_len * mask_ratio))
            positions = torch.randperm(seq_len)[:n_mask] + 1  # +1 for BOS
            mask[i, positions] = True

        # BERT 방식: 80% MASK, 10% random, 10% keep
        rand = torch.rand_like(batch_tokens, dtype=torch.float)
        mask_token_mask = mask & (rand < 0.8)
        random_token_mask = mask & (rand >= 0.8) & (rand < 0.9)

        batch_tokens[mask_token_mask] = self.alphabet.mask_idx
        if random_token_mask.any():
            random_tokens = torch.randint(
                4, len(self.alphabet) - 1,  # standard token range
                (random_token_mask.sum(),)
            )
            batch_tokens[random_token_mask] = random_tokens.to(self.device)

        # Forward pass
        output = self.model(batch_tokens)
        logits = output["logits"]

        # 마스크된 위치만 손실 계산
        loss_fn = torch.nn.CrossEntropyLoss()
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        if masked_logits.shape[0] == 0:
            return None

        loss = loss_fn(masked_logits, masked_labels)
        return loss

    def save_finetuned(self, name: str = "peptone_finetuned"):
        """Fine-tuned 모델 저장"""
        import torch

        save_path = self.save_dir / f"{name}_{self.model_name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "training_history": self.training_history,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, str(save_path))

        logger.info(f"Fine-tuned model saved: {save_path}")
        return str(save_path)

    def load_finetuned(self, name: str = "peptone_finetuned") -> bool:
        """Fine-tuned 모델 로드"""
        import torch

        save_path = self.save_dir / f"{name}_{self.model_name}.pt"
        if not save_path.exists():
            return False

        if self.model is None:
            self.load_base_model()

        checkpoint = torch.load(str(save_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.model.eval()

        logger.info(f"Fine-tuned model loaded: {save_path}")
        return True

    def list_finetuned_models(self) -> List[Dict]:
        """저장된 Fine-tuned 모델 목록"""
        import torch

        models = []
        for f in self.save_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(str(f), map_location="cpu")
                models.append({
                    "name": f.stem,
                    "path": str(f),
                    "model_name": checkpoint.get("model_name", "unknown"),
                    "timestamp": checkpoint.get("timestamp", "unknown"),
                    "epochs": len(checkpoint.get("training_history", [])),
                })
            except Exception:
                pass
        return models


class ModelComparator:
    """
    범용 ESM-2 vs 펩톤 특화 ESM-2 비교 분석

    동일한 서열에 대해 두 모델의 차이를 정량적으로 비교
    """

    def __init__(self, base_embedder, finetuned_embedder):
        """
        Args:
            base_embedder: PLMEmbedder (범용 모델)
            finetuned_embedder: PLMEmbedder (Fine-tuned 모델)
        """
        self.base = base_embedder
        self.finetuned = finetuned_embedder

    def compare_perplexity(self, sequences: List[str]) -> Dict:
        """
        서열별 Perplexity 비교 (낮을수록 모델이 서열을 잘 이해함)
        """
        results = []
        for seq in sequences:
            base_ppl = self._compute_perplexity(self.base, seq)
            ft_ppl = self._compute_perplexity(self.finetuned, seq)

            improvement = ((base_ppl - ft_ppl) / base_ppl * 100) if base_ppl > 0 else 0

            results.append({
                "sequence": seq,
                "length": len(seq),
                "base_perplexity": round(base_ppl, 2),
                "finetuned_perplexity": round(ft_ppl, 2),
                "improvement_pct": round(improvement, 1),
            })

        avg_improvement = np.mean([r["improvement_pct"] for r in results])

        return {
            "per_sequence": results,
            "avg_base_ppl": round(np.mean([r["base_perplexity"] for r in results]), 2),
            "avg_ft_ppl": round(np.mean([r["finetuned_perplexity"] for r in results]), 2),
            "avg_improvement_pct": round(avg_improvement, 1),
        }

    def compare_embeddings(self, sequences: List[str]) -> Dict:
        """
        임베딩 차이 분석: 두 모델의 임베딩이 얼마나 다른지
        """
        base_embs = []
        ft_embs = []

        for seq in sequences:
            base_embs.append(self.base.get_sequence_embedding(seq))
            ft_embs.append(self.finetuned.get_sequence_embedding(seq))

        base_embs = np.array(base_embs)
        ft_embs = np.array(ft_embs)

        # 코사인 유사도 (두 모델의 임베딩 간)
        cosine_sims = []
        for b, f in zip(base_embs, ft_embs):
            sim = np.dot(b, f) / (np.linalg.norm(b) * np.linalg.norm(f) + 1e-10)
            cosine_sims.append(float(sim))

        # L2 거리
        l2_dists = [float(np.linalg.norm(b - f)) for b, f in zip(base_embs, ft_embs)]

        return {
            "per_sequence": [
                {
                    "sequence": seq,
                    "cosine_similarity": round(sim, 4),
                    "l2_distance": round(dist, 4),
                    "shift_magnitude": "large" if sim < 0.9 else "medium" if sim < 0.95 else "small"
                }
                for seq, sim, dist in zip(sequences, cosine_sims, l2_dists)
            ],
            "avg_cosine_similarity": round(np.mean(cosine_sims), 4),
            "avg_l2_distance": round(np.mean(l2_dists), 4),
            "base_embeddings": base_embs,
            "finetuned_embeddings": ft_embs,
        }

    def compare_mutation_scores(self, sequence: str, mutations: List[str]) -> Dict:
        """
        변이 예측 비교: 동일한 변이에 대해 두 모델의 score 비교
        """
        base_scores = self.base.zero_shot_score(sequence, mutations)
        ft_scores = self.finetuned.zero_shot_score(sequence, mutations)

        comparisons = []
        for mut in mutations:
            base_data = base_scores.get(mut, {})
            ft_data = ft_scores.get(mut, {})

            if "error" in base_data or "error" in ft_data:
                continue

            comparisons.append({
                "mutation": mut,
                "base_score": base_data.get("score", 0),
                "finetuned_score": ft_data.get("score", 0),
                "base_effect": base_data.get("effect", "unknown"),
                "finetuned_effect": ft_data.get("effect", "unknown"),
                "score_shift": round(ft_data.get("score", 0) - base_data.get("score", 0), 4),
                "effect_changed": base_data.get("effect") != ft_data.get("effect"),
            })

        return {
            "comparisons": comparisons,
            "n_effect_changes": sum(1 for c in comparisons if c["effect_changed"]),
            "avg_score_shift": round(
                np.mean([c["score_shift"] for c in comparisons]), 4
            ) if comparisons else 0,
        }

    def _compute_perplexity(self, embedder, sequence: str) -> float:
        """단일 서열의 pseudo-perplexity 계산"""
        import torch

        embedder.load_model()
        data = [("protein", sequence)]
        _, _, batch_tokens = embedder.batch_converter(data)
        batch_tokens = batch_tokens.to(embedder.device)

        total_nll = 0.0
        n_tokens = 0

        with torch.no_grad():
            for i in range(len(sequence)):
                masked = batch_tokens.clone()
                masked[0, i + 1] = embedder.alphabet.mask_idx

                logits = embedder.model(masked)["logits"]
                probs = torch.softmax(logits[0, i + 1], dim=-1)

                aa_idx = embedder.alphabet.get_idx(sequence[i])
                prob = probs[aa_idx].item()
                total_nll += -np.log(prob + 1e-10)
                n_tokens += 1

        return float(np.exp(total_nll / max(n_tokens, 1)))
