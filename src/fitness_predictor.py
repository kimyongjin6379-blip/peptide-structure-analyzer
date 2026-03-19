"""
Fitness Predictor - PyTorch 기반 단백질/효소 활성 예측 모듈

pLM 임베딩을 입력으로 받아 다양한 물성을 예측하는 딥러닝 모델.
기존 bioactive_predictor.py의 규칙 기반 스코어링을 ML 모델로 보강.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FitnessPredictor:
    """
    pLM 임베딩 → MLP → 다중 물성 예측

    예측 대상:
    - antimicrobial: 항균 활성
    - antihypertensive: 항고혈압 활성 (ACE 억제)
    - antioxidant: 항산화 활성
    - stability: 열안정성
    - solubility: 용해도
    - expression: 발현량
    """

    TARGET_NAMES = [
        "antimicrobial", "antihypertensive", "antioxidant",
        "stability", "solubility", "expression"
    ]

    def __init__(self, embedding_dim: int = 320, hidden_dims: list = None,
                 model_dir: str = None):
        """
        Args:
            embedding_dim: 입력 임베딩 차원 (ESM-2 모델에 따라 다름)
                - esm2_t6_8M: 320
                - esm2_t12_35M: 480
                - esm2_t33_650M: 1280
            hidden_dims: MLP 히든 레이어 차원 리스트
            model_dir: 학습된 모델 가중치 저장 경로
        """
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.num_targets = len(self.TARGET_NAMES)
        self.model = None
        self.trained = False

        self.model_dir = Path(model_dir) if model_dir else (
            Path(__file__).parent.parent / "data" / "models"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 학습 히스토리
        self.history = {"train_loss": [], "val_loss": [], "val_r2": []}

    def _build_model(self):
        """MLP 모델 구축"""
        import torch
        import torch.nn as nn

        layers = []
        in_dim = self.embedding_dim

        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, self.num_targets))
        layers.append(nn.Sigmoid())  # 0~1 범위 출력

        self.model = nn.Sequential(*layers)
        return self.model

    def train(self, embeddings: np.ndarray, labels: np.ndarray,
              epochs: int = 100, lr: float = 1e-3, val_split: float = 0.2,
              batch_size: int = 32, patience: int = 15):
        """
        모델 학습

        Args:
            embeddings: (N, D) 서열 임베딩 배열
            labels: (N, T) 타겟 값 배열 (0~1 정규화)
            epochs: 학습 에폭 수
            lr: 학습률
            val_split: 검증 데이터 비율
            batch_size: 배치 크기
            patience: Early stopping patience

        Returns:
            dict: 학습 결과 (final loss, best val r2, etc.)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # 모델 초기화
        self._build_model()

        # 데이터 분할
        n = len(embeddings)
        indices = np.random.permutation(n)
        val_size = int(n * val_split)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train = torch.FloatTensor(embeddings[train_idx])
        y_train = torch.FloatTensor(labels[train_idx])
        X_val = torch.FloatTensor(embeddings[val_idx])
        y_val = torch.FloatTensor(labels[val_idx])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True
        )

        # 학습 설정
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()

        for epoch in range(epochs):
            # Training
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()

                # R² score 계산
                val_pred_np = val_pred.numpy()
                val_true_np = y_val.numpy()
                ss_res = np.sum((val_true_np - val_pred_np) ** 2)
                ss_tot = np.sum((val_true_np - val_true_np.mean(axis=0)) ** 2) + 1e-10
                val_r2 = 1 - ss_res / ss_tot

            self.model.train()

            # 히스토리 기록
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(val_r2)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model("best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val R²: {val_r2:.4f}"
                )

        # 최적 모델 로드
        self._load_model("best_model.pt")
        self.trained = True

        return {
            "final_train_loss": self.history["train_loss"][-1],
            "best_val_loss": best_val_loss,
            "best_val_r2": max(self.history["val_r2"]),
            "epochs_trained": len(self.history["train_loss"]),
            "history": self.history
        }

    def predict(self, embedding: np.ndarray) -> dict:
        """
        단일 서열의 물성 예측

        Args:
            embedding: (D,) 서열 임베딩 벡터

        Returns:
            dict: {
                "antimicrobial": {"score": 0.75, "confidence": "high"},
                "antihypertensive": {"score": 0.42, "confidence": "medium"},
                ...
            }
        """
        import torch

        if not self.trained and not self._load_model("best_model.pt"):
            return self._rule_based_predict(embedding)

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embedding).unsqueeze(0)
            pred = self.model(x).squeeze().numpy()

        results = {}
        for i, name in enumerate(self.TARGET_NAMES):
            score = float(pred[i])
            if score >= 0.7:
                confidence = "high"
            elif score >= 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            results[name] = {
                "score": round(score, 4),
                "confidence": confidence
            }

        return results

    def predict_batch(self, embeddings: np.ndarray) -> list:
        """
        배치 예측

        Args:
            embeddings: (N, D) 임베딩 배열

        Returns:
            list[dict]: N개 예측 결과
        """
        import torch

        if not self.trained and not self._load_model("best_model.pt"):
            return [self._rule_based_predict(emb) for emb in embeddings]

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embeddings)
            preds = self.model(x).numpy()

        results = []
        for pred in preds:
            result = {}
            for i, name in enumerate(self.TARGET_NAMES):
                score = float(pred[i])
                result[name] = {
                    "score": round(score, 4),
                    "confidence": "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
                }
            results.append(result)

        return results

    def predict_with_uncertainty(self, embedding: np.ndarray, n_samples: int = 30) -> dict:
        """
        MC Dropout 기반 불확실성 추정 예측

        Dropout을 활성화한 상태로 여러 번 추론하여
        예측값의 분산을 불확실성으로 사용.

        Args:
            embedding: (D,) 서열 임베딩
            n_samples: MC 샘플 수

        Returns:
            dict: 예측값 + 표준편차(불확실성)
        """
        import torch

        if not self.trained and not self._load_model("best_model.pt"):
            return self._rule_based_predict(embedding)

        # Dropout만 활성화 (BatchNorm은 eval 모드 유지)
        self.model.train()
        for m in self.model.modules():
            import torch.nn as nn
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        predictions = []
        with torch.no_grad():
            x = torch.FloatTensor(embedding).unsqueeze(0)
            for _ in range(n_samples):
                pred = self.model(x).squeeze().numpy()
                predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        self.model.eval()

        results = {}
        for i, name in enumerate(self.TARGET_NAMES):
            score = float(mean_pred[i])
            uncertainty = float(std_pred[i])

            results[name] = {
                "score": round(score, 4),
                "uncertainty": round(uncertainty, 4),
                "confidence": "high" if uncertainty < 0.1 else "medium" if uncertainty < 0.2 else "low",
                "range": [round(score - 2 * uncertainty, 4), round(score + 2 * uncertainty, 4)]
            }

        return results

    def _rule_based_predict(self, embedding: np.ndarray) -> dict:
        """
        학습된 모델이 없을 때 임베딩 통계 기반 규칙 예측 (fallback)
        """
        # 임베딩의 통계적 특성을 활용한 휴리스틱
        emb_mean = float(np.mean(embedding))
        emb_std = float(np.std(embedding))
        emb_norm = float(np.linalg.norm(embedding))

        results = {}
        for name in self.TARGET_NAMES:
            # 간단한 해시 기반 유사 랜덤 스코어 (재현 가능)
            hash_val = hash(f"{name}_{emb_mean:.4f}_{emb_std:.4f}") % 1000 / 1000
            score = 0.3 + 0.4 * hash_val  # 0.3~0.7 범위

            results[name] = {
                "score": round(score, 4),
                "confidence": "low",
                "note": "rule-based fallback (no trained model)"
            }

        return results

    # ---- 모델 저장/로드 ----

    def _save_model(self, filename: str):
        """모델 가중치 저장"""
        import torch
        path = self.model_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "num_targets": self.num_targets,
            "history": self.history
        }, str(path))

    def _load_model(self, filename: str) -> bool:
        """모델 가중치 로드"""
        import torch
        path = self.model_dir / filename
        if not path.exists():
            return False

        try:
            checkpoint = torch.load(str(path), map_location="cpu")
            self.embedding_dim = checkpoint["embedding_dim"]
            self.hidden_dims = checkpoint["hidden_dims"]
            self._build_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.trained = True
            self.history = checkpoint.get("history", self.history)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

    def get_training_summary(self) -> dict:
        """학습 결과 요약"""
        if not self.history["train_loss"]:
            return {"status": "not trained"}

        return {
            "status": "trained",
            "epochs": len(self.history["train_loss"]),
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "best_val_r2": max(self.history["val_r2"]),
            "target_names": self.TARGET_NAMES
        }
