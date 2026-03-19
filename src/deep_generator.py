"""
Deep Generator - 딥러닝 기반 펩타이드 서열 생성 모듈

기존 Markov Chain (sequence_predictor.py) 을 보강하는 DL 생성 모델:
1. VAE (Variational Autoencoder) - 잠재 공간 탐색 기반 생성
2. ProtGPT2 - 사전학습 LM 기반 자가회귀 생성
3. ESM-2 Masked Generation - pLM 기반 마스크 채우기 생성
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 표준 아미노산
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}


class PeptideVAE:
    """
    VAE 기반 펩타이드 서열 생성기

    Encoder: 서열 → 잠재 벡터 z
    Decoder: 잠재 벡터 z → 서열

    장점:
    - 잠재 공간에서 연속적 탐색 가능
    - 두 서열 사이 보간(interpolation) 가능
    - 조건부 생성 (CVAE) 확장 가능
    """

    def __init__(self, max_len: int = 50, latent_dim: int = 32,
                 hidden_dim: int = 128):
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(AMINO_ACIDS) + 1  # +1 for padding
        self.model = None
        self.encoder = None
        self.decoder = None
        self.trained = False

        self.model_dir = Path(__file__).parent.parent / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self):
        """VAE 모델 구축"""
        import torch
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self, vocab_size, max_len, hidden_dim, latent_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                                    bidirectional=True)
                self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

            def forward(self, x):
                embedded = self.embedding(x)
                _, (h, _) = self.lstm(embedded)
                h = torch.cat([h[0], h[1]], dim=-1)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                return mu, logvar

        class Decoder(nn.Module):
            def __init__(self, vocab_size, max_len, hidden_dim, latent_dim):
                super().__init__()
                self.max_len = max_len
                self.fc = nn.Linear(latent_dim, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.output = nn.Linear(hidden_dim, vocab_size)
                self.hidden_dim = hidden_dim

            def forward(self, z):
                h = self.fc(z).unsqueeze(0)
                c = torch.zeros_like(h)
                # 입력을 z에서 반복 생성
                repeated = self.fc(z).unsqueeze(1).repeat(1, self.max_len, 1)
                output, _ = self.lstm(repeated, (h, c))
                logits = self.output(output)
                return logits

        self.encoder = Encoder(self.vocab_size, self.max_len,
                               self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.vocab_size, self.max_len,
                               self.hidden_dim, self.latent_dim)

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """서열 → 정수 인코딩"""
        encoded = np.zeros(self.max_len, dtype=np.int64)
        for i, aa in enumerate(sequence[:self.max_len]):
            encoded[i] = AA_TO_IDX.get(aa, 0) + 1  # +1 (0 = padding)
        return encoded

    def _decode_indices(self, indices: np.ndarray) -> str:
        """정수 인코딩 → 서열"""
        sequence = ""
        for idx in indices:
            idx = int(idx) - 1  # -1 (padding offset)
            if idx < 0 or idx >= len(AMINO_ACIDS):
                break
            sequence += IDX_TO_AA[idx]
        return sequence

    def train(self, sequences: list, epochs: int = 200, lr: float = 1e-3,
              batch_size: int = 32, kl_weight: float = 0.1):
        """
        VAE 학습

        Args:
            sequences: 학습용 아미노산 서열 리스트
            epochs: 학습 에폭
            lr: 학습률
            batch_size: 배치 크기
            kl_weight: KL divergence 가중치 (β-VAE)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._build_model()

        # 데이터 준비
        encoded = np.array([self._encode_sequence(seq) for seq in sequences])
        dataset = TensorDataset(torch.LongTensor(encoded))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 옵티마이저
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding 무시

        self.encoder.train()
        self.decoder.train()

        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for (batch,) in loader:
                optimizer.zero_grad()

                # Encode
                mu, logvar = self.encoder(batch)

                # Reparameterization trick
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                # Decode
                logits = self.decoder(z)

                # Loss
                recon_loss = criterion(
                    logits.view(-1, self.vocab_size),
                    batch.view(-1)
                )
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)

                loss = recon_loss + kl_weight * kl_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            if (epoch + 1) % 20 == 0:
                n_batches = len(loader)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {total_loss / n_batches:.4f} | "
                    f"Recon: {total_recon / n_batches:.4f} | "
                    f"KL: {total_kl / n_batches:.4f}"
                )

        self.trained = True
        self._save_model()
        logger.info("VAE training complete")

    def generate(self, n: int = 10, temperature: float = 1.0,
                 max_length: Optional[int] = None) -> list:
        """
        잠재 공간에서 샘플링하여 새 서열 생성

        Args:
            n: 생성할 서열 수
            temperature: 샘플링 온도 (높을수록 다양, 낮을수록 보수적)
            max_length: 최대 서열 길이

        Returns:
            list[dict]: [{"sequence": "ACDEF...", "length": 10}, ...]
        """
        import torch

        if not self.trained:
            self._load_model()

        if self.decoder is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")

        self.decoder.eval()
        results = []

        with torch.no_grad():
            # 표준 정규 분포에서 z 샘플링
            z = torch.randn(n, self.latent_dim) * temperature
            logits = self.decoder(z)

            # 확률적 샘플링
            probs = torch.softmax(logits / temperature, dim=-1)

            for i in range(n):
                indices = torch.multinomial(probs[i], 1).squeeze(-1).numpy()
                sequence = self._decode_indices(indices)

                if max_length and len(sequence) > max_length:
                    sequence = sequence[:max_length]

                if len(sequence) >= 3:  # 최소 3잔기
                    results.append({
                        "sequence": sequence,
                        "length": len(sequence),
                        "method": "VAE"
                    })

        return results

    def interpolate(self, seq1: str, seq2: str, steps: int = 10) -> list:
        """
        두 서열 사이 잠재 공간 보간

        seq1의 잠재 벡터에서 seq2의 잠재 벡터까지
        선형 보간하여 중간 서열들을 생성.
        """
        import torch

        if not self.trained:
            self._load_model()

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # 두 서열의 잠재 벡터 추출
            x1 = torch.LongTensor(self._encode_sequence(seq1)).unsqueeze(0)
            x2 = torch.LongTensor(self._encode_sequence(seq2)).unsqueeze(0)

            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)

            results = []
            for i in range(steps):
                alpha = i / (steps - 1)
                z = mu1 * (1 - alpha) + mu2 * alpha

                logits = self.decoder(z)
                indices = logits.argmax(dim=-1).squeeze().numpy()
                sequence = self._decode_indices(indices)

                results.append({
                    "sequence": sequence,
                    "alpha": round(alpha, 2),
                    "method": "VAE_interpolation"
                })

        return results

    def _save_model(self):
        import torch
        path = self.model_dir / "vae_model.pt"
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "max_len": self.max_len,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim
        }, str(path))

    def _load_model(self) -> bool:
        import torch
        path = self.model_dir / "vae_model.pt"
        if not path.exists():
            return False
        try:
            checkpoint = torch.load(str(path), map_location="cpu")
            self.max_len = checkpoint["max_len"]
            self.latent_dim = checkpoint["latent_dim"]
            self.hidden_dim = checkpoint["hidden_dim"]
            self._build_model()
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.trained = True
            return True
        except Exception as e:
            logger.warning(f"VAE model load failed: {e}")
            return False


class ProtGPT2Generator:
    """
    ProtGPT2 기반 자가회귀 서열 생성

    Hugging Face에서 사전학습된 ProtGPT2를 로드하여
    자연스러운 단백질 서열을 생성.
    """

    def __init__(self, model_name: str = "nferruz/ProtGPT2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """ProtGPT2 모델 로딩"""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading ProtGPT2...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("ProtGPT2 loaded successfully")

        except ImportError:
            raise ImportError(
                "transformers 패키지가 필요합니다. "
                "`pip install transformers` 으로 설치하세요."
            )

    def generate(self, prompt: str = "", max_length: int = 100,
                 n: int = 10, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.2) -> list:
        """
        ProtGPT2로 서열 생성

        Args:
            prompt: 시작 서열 (빈 문자열이면 처음부터 생성)
            max_length: 최대 생성 길이
            n: 생성할 서열 수
            temperature: 샘플링 온도
            top_k: Top-K 샘플링
            top_p: Nucleus 샘플링
            repetition_penalty: 반복 억제

        Returns:
            list[dict]: 생성된 서열 정보
        """
        self.load_model()

        import torch

        # 프롬프트가 비어있으면 시작 토큰만
        if prompt:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
        else:
            # BOS 토큰으로 시작
            input_ids = torch.tensor([[self.tokenizer.bos_token_id or 0]])

        results = []

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=n,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        for i, output in enumerate(outputs):
            generated = self.tokenizer.decode(output, skip_special_tokens=True)

            # 유효한 아미노산만 필터링
            clean_seq = "".join(c for c in generated if c in AMINO_ACIDS)

            if len(clean_seq) >= 3:
                results.append({
                    "sequence": clean_seq,
                    "length": len(clean_seq),
                    "method": "ProtGPT2",
                    "raw_output": generated[:100]  # 디버그용
                })

        return results

    def score_sequence(self, sequence: str) -> float:
        """
        서열의 log-likelihood 계산

        ProtGPT2가 보기에 "자연스러운" 서열일수록 높은 점수.

        Returns:
            float: 평균 log-likelihood (높을수록 좋음)
        """
        self.load_model()

        import torch

        inputs = self.tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # Negative log-likelihood → 부호 반전
            nll = outputs.loss.item()

        return round(-nll, 4)


class ESMMaskedGenerator:
    """
    ESM-2 기반 마스크 채우기(Masked Infilling) 서열 생성

    서열의 특정 위치를 마스크하고 ESM-2가 채우도록 하여
    진화적으로 타당한 변이체를 생성.
    """

    def __init__(self, plm_embedder=None):
        """
        Args:
            plm_embedder: PLMEmbedder 인스턴스 (이미 있으면 재사용)
        """
        self.plm_embedder = plm_embedder

    def _ensure_model(self):
        if self.plm_embedder is None:
            from plm_embedder import PLMEmbedder
            self.plm_embedder = PLMEmbedder(model_name="esm2_t6_8M")
        self.plm_embedder.load_model()

    def generate_variants(self, sequence: str, positions: list = None,
                          n_per_position: int = 5) -> list:
        """
        특정 위치를 마스크하여 ESM-2가 제안하는 변이체 생성

        Args:
            sequence: 원본 서열
            positions: 마스크할 위치 (1-based). None이면 모든 위치
            n_per_position: 위치당 생성할 변이체 수

        Returns:
            list[dict]: 변이체 정보
        """
        self._ensure_model()

        import torch

        model = self.plm_embedder.model
        alphabet = self.plm_embedder.alphabet
        batch_converter = self.plm_embedder.batch_converter

        if positions is None:
            positions = list(range(1, len(sequence) + 1))

        results = []

        data = [("protein", sequence)]
        _, _, batch_tokens = batch_converter(data)

        for pos in positions:
            if pos < 1 or pos > len(sequence):
                continue

            # 해당 위치 마스크
            masked_tokens = batch_tokens.clone()
            masked_tokens[0, pos] = alphabet.mask_idx  # pos는 이미 1-based (BOS offset)

            with torch.no_grad():
                logits = model(masked_tokens)["logits"]
                probs = torch.softmax(logits[0, pos], dim=-1)

            # Top-K 아미노산 추출
            top_probs, top_indices = probs.topk(n_per_position + 1)

            for i in range(len(top_indices)):
                aa = alphabet.get_tok(top_indices[i].item())
                prob = top_probs[i].item()

                if aa not in AMINO_ACIDS:
                    continue
                if aa == sequence[pos - 1]:
                    continue  # 원본과 동일한 건 스킵

                # 변이체 서열 구성
                variant_seq = sequence[:pos - 1] + aa + sequence[pos:]
                mutation_str = f"{sequence[pos - 1]}{pos}{aa}"

                results.append({
                    "sequence": variant_seq,
                    "mutation": mutation_str,
                    "position": pos,
                    "wt_aa": sequence[pos - 1],
                    "mt_aa": aa,
                    "probability": round(prob, 4),
                    "method": "ESM2_masked"
                })

                if len([r for r in results if r["position"] == pos]) >= n_per_position:
                    break

        # 확률 내림차순 정렬
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def iterative_refinement(self, sequence: str, n_iterations: int = 5,
                             n_mutations_per_iter: int = 2) -> list:
        """
        반복적 마스크 채우기로 서열 최적화

        매 반복마다:
        1. 랜덤 위치 마스크
        2. ESM-2가 최적 아미노산 제안
        3. 서열 업데이트
        → 점진적으로 진화적 타당성이 높은 서열로 수렴

        Args:
            sequence: 시작 서열
            n_iterations: 반복 횟수
            n_mutations_per_iter: 반복당 변이 수

        Returns:
            list[dict]: 반복별 서열 변화 기록
        """
        self._ensure_model()

        import torch

        current_seq = sequence
        trajectory = [{
            "iteration": 0,
            "sequence": current_seq,
            "mutations": [],
            "method": "ESM2_refinement"
        }]

        for iteration in range(1, n_iterations + 1):
            # 랜덤 위치 선택
            positions = np.random.choice(
                range(1, len(current_seq) + 1),
                size=min(n_mutations_per_iter, len(current_seq)),
                replace=False
            ).tolist()

            mutations = []
            for pos in sorted(positions):
                variants = self.generate_variants(
                    current_seq, positions=[pos], n_per_position=1
                )
                if variants:
                    best = variants[0]
                    current_seq = best["sequence"]
                    mutations.append(best["mutation"])

            trajectory.append({
                "iteration": iteration,
                "sequence": current_seq,
                "mutations": mutations,
                "method": "ESM2_refinement"
            })

        return trajectory


class DeepGeneratorManager:
    """
    모든 생성 모델을 통합 관리하는 매니저

    Streamlit UI에서 이 클래스 하나로 모든 생성 방법에 접근.
    """

    def __init__(self, plm_embedder=None):
        self._vae = None
        self._protgpt2 = None
        self._esm_gen = None
        self._plm_embedder = plm_embedder

    @property
    def vae(self) -> PeptideVAE:
        if self._vae is None:
            self._vae = PeptideVAE()
        return self._vae

    @property
    def protgpt2(self) -> ProtGPT2Generator:
        if self._protgpt2 is None:
            self._protgpt2 = ProtGPT2Generator()
        return self._protgpt2

    @property
    def esm_generator(self) -> ESMMaskedGenerator:
        if self._esm_gen is None:
            self._esm_gen = ESMMaskedGenerator(plm_embedder=self._plm_embedder)
        return self._esm_gen

    def generate(self, method: str, **kwargs) -> list:
        """
        통합 생성 인터페이스

        Args:
            method: "vae", "protgpt2", "esm_masked", "esm_refinement"
            **kwargs: 각 메서드별 인자

        Returns:
            list[dict]: 생성된 서열 리스트
        """
        if method == "vae":
            return self.vae.generate(**kwargs)
        elif method == "protgpt2":
            return self.protgpt2.generate(**kwargs)
        elif method == "esm_masked":
            return self.esm_generator.generate_variants(**kwargs)
        elif method == "esm_refinement":
            return self.esm_generator.iterative_refinement(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. "
                           f"Available: vae, protgpt2, esm_masked, esm_refinement")

    def get_available_methods(self) -> dict:
        """사용 가능한 생성 방법 목록"""
        return {
            "vae": {
                "name": "VAE (Variational Autoencoder)",
                "description": "잠재 공간 탐색 기반 서열 생성. 학습 필요.",
                "requires_training": True,
                "speed": "fast (학습 후)"
            },
            "protgpt2": {
                "name": "ProtGPT2",
                "description": "사전학습된 단백질 언어 모델 기반 자가회귀 생성.",
                "requires_training": False,
                "speed": "medium (~5초/10서열)"
            },
            "esm_masked": {
                "name": "ESM-2 Masked Generation",
                "description": "ESM-2 마스크 채우기로 진화적으로 타당한 변이체 생성.",
                "requires_training": False,
                "speed": "fast (~1초/위치)"
            },
            "esm_refinement": {
                "name": "ESM-2 Iterative Refinement",
                "description": "반복적 마스크 채우기로 서열 점진 최적화.",
                "requires_training": False,
                "speed": "medium (~10초/5반복)"
            }
        }
