"""
AI 심층 분석 페이지

Railway 환경에서 활용 가능한 딥러닝 기반 분석 기능:
1. ESM-2 pLM 임베딩 분석
2. Zero-shot 변이 효과 예측
3. 딥러닝 기반 서열 생성 (VAE / ProtGPT2 / ESM-2 Masked)
4. ML 기반 활성 예측
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Analysis", page_icon="🤖", layout="wide")


# ---- 모델 가용성 체크 ----
def check_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


def check_esm_available():
    try:
        import esm
        return True
    except ImportError:
        return False


def check_transformers_available():
    try:
        import transformers
        return True
    except ImportError:
        return False


TORCH_OK = check_torch_available()
ESM_OK = check_esm_available()
TRANSFORMERS_OK = check_transformers_available()


# ---- 캐싱 ----
@st.cache_resource
def get_plm_embedder(model_name="esm2_t6_8M", finetuned_name=None):
    from plm_embedder import PLMEmbedder
    embedder = PLMEmbedder(model_name=model_name)
    embedder.load_model()

    # Fine-tuned 모델 가중치 로드
    if finetuned_name:
        try:
            from plm_finetuner import PLMFineTuner
            finetuner = PLMFineTuner(model_name=model_name)
            if finetuner.load_finetuned(name=finetuned_name):
                embedder.model = finetuner.model
                embedder.model.eval()
        except Exception:
            pass  # Fine-tuned 모델 없으면 범용 모델 사용

    return embedder


@st.cache_resource
def get_deep_generator():
    from deep_generator import DeepGeneratorManager
    return DeepGeneratorManager()


# ---- 메인 ----
def main():
    st.markdown("# 🤖 AI 심층 분석")
    st.markdown("ESM-2 Protein Language Model과 딥러닝 기반 고급 분석 도구")

    # 환경 체크
    with st.sidebar:
        st.markdown("### ⚙️ 시스템 상태")
        st.write(f"PyTorch: {'✅' if TORCH_OK else '❌'}")
        st.write(f"ESM: {'✅' if ESM_OK else '❌'}")
        st.write(f"Transformers: {'✅' if TRANSFORMERS_OK else '❌'}")

        if not TORCH_OK:
            st.warning("⚠️ PyTorch가 설치되지 않았습니다. Railway 배포 환경에서 사용하세요.")

        st.markdown("---")
        st.markdown("### 🧠 ESM-2 모델 선택")
        model_choice = st.selectbox(
            "모델 크기",
            ["esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M"],
            index=0,
            help="큰 모델일수록 정확하지만 느림"
        )

        # Fine-tuned 모델 선택
        st.markdown("### 🎯 Fine-tuned 모델")
        finetuned_name = None
        try:
            from plm_finetuner import PLMFineTuner
            ft_check = PLMFineTuner(model_name=model_choice)
            ft_models = ft_check.list_finetuned_models()
            if ft_models:
                ft_options = ["범용 ESM-2 (기본)"] + [m['name'] for m in ft_models]
                ft_selected = st.selectbox("사용할 모델", ft_options, index=0)
                if ft_selected != "범용 ESM-2 (기본)":
                    finetuned_name = ft_selected.replace(f"_{model_choice}", "")
                    st.success(f"🎯 펩톤 특화 모델 사용 중")
                else:
                    st.info("범용 모델 사용 중")
            else:
                st.caption("Fine-tuned 모델 없음 (7번 페이지에서 학습)")
        except Exception:
            st.caption("Fine-tuned 모델 확인 불가")

    if not TORCH_OK:
        st.error(
            "이 페이지는 **Railway 배포 환경**에서만 사용 가능합니다.\n\n"
            "PyTorch, ESM-2 등 무거운 ML 패키지가 필요합니다.\n\n"
            "현재 Streamlit Cloud 환경에서는 기존 1~5번 페이지를 이용해주세요."
        )
        st.info(
            "💡 **Railway 배포 방법**\n\n"
            "```bash\n"
            "# GitHub에 push 후 Railway에서 Docker 배포\n"
            "railway up\n"
            "```"
        )
        return

    # ---- 다른 페이지에서 전송된 서열 확인 ----
    transferred_seq = st.session_state.get('ai_input_sequence', None)
    transferred_batch = st.session_state.get('ai_batch_sequences', None)
    transfer_source = st.session_state.get('ai_batch_source', '')

    if transferred_seq or transferred_batch:
        st.markdown("---")
        st.markdown("### 📨 전송된 서열")
        if transferred_seq:
            st.success(f"**단일 서열**: `{transferred_seq[:60]}{'...' if len(transferred_seq) > 60 else ''}`")
        if transferred_batch:
            st.info(f"**배치 서열**: {len(transferred_batch)}개 ({transfer_source})")

        col_clear1, col_clear2 = st.columns([1, 4])
        with col_clear1:
            if st.button("🗑️ 전송 데이터 초기화", key="clear_transfer"):
                for key in ['ai_input_sequence', 'ai_target_tab', 'ai_batch_sequences', 'ai_batch_source']:
                    st.session_state.pop(key, None)
                st.rerun()
        st.markdown("---")

    # ---- 탭 구성 ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧬 서열 임베딩", "🔬 변이 예측", "🧪 DL 서열 생성", "📊 활성 예측",
        "📦 배치 분석"
    ])

    # ============================================================
    # 탭 1: 서열 임베딩 분석
    # ============================================================
    with tab1:
        st.markdown("## ESM-2 서열 임베딩 분석")
        st.markdown(
            "단백질 언어 모델이 학습한 **진화적 표현(representation)**을 추출하여 "
            "서열의 기능적 특성을 분석합니다."
        )

        # 전송된 서열이 있으면 기본값으로 사용
        default_seq = transferred_seq if transferred_seq else "ACDEFGHIKLMNPQRSTVWY"

        col1, col2 = st.columns([2, 1])
        with col1:
            input_sequence = st.text_area(
                "아미노산 서열 입력",
                value=default_seq,
                height=100,
                help="표준 20종 아미노산 문자만 입력. 2번/3번 페이지에서 전송된 서열이 자동 입력됩니다."
            )
        with col2:
            st.markdown("**입력 정보**")
            clean_seq = "".join(c for c in input_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            st.write(f"서열 길이: {len(clean_seq)}")
            st.write(f"선택 모델: {model_choice}")
            if transferred_seq:
                st.caption("📨 2/3번 페이지에서 전송됨")

        if st.button("🚀 임베딩 추출", key="embed_btn"):
            if len(clean_seq) < 3:
                st.error("서열이 너무 짧습니다 (최소 3잔기)")
            else:
                with st.spinner(f"ESM-2 ({model_choice}) 모델 로딩 및 임베딩 추출 중..."):
                    try:
                        embedder = get_plm_embedder(model_choice, finetuned_name)
                        embedding = embedder.get_embedding(clean_seq)
                        seq_embedding = embedding.mean(axis=0)

                        st.success(f"✅ 임베딩 추출 완료! 차원: {embedding.shape}")

                        # 임베딩 히트맵
                        st.markdown("### 잔기별 임베딩 히트맵")
                        fig = go.Figure(data=go.Heatmap(
                            z=embedding[:, :50],  # 처음 50차원만 표시
                            x=[f"dim_{i}" for i in range(50)],
                            y=[f"{aa}{i+1}" for i, aa in enumerate(clean_seq)],
                            colorscale="Viridis"
                        ))
                        fig.update_layout(
                            height=max(300, len(clean_seq) * 20),
                            title="잔기별 ESM-2 임베딩 (처음 50차원)"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 잔기 중요도
                        st.markdown("### 잔기별 중요도 분석")
                        with st.spinner("잔기 중요도 계산 중..."):
                            importance = embedder.get_residue_importance(clean_seq)

                        imp_df = pd.DataFrame([
                            {"Position": pos, "AA": clean_seq[pos-1],
                             "Importance": score}
                            for pos, score in importance.items()
                        ])

                        fig_imp = px.bar(
                            imp_df, x="Position", y="Importance",
                            color="Importance",
                            color_continuous_scale="RdYlGn_r",
                            hover_data=["AA"],
                            title="잔기별 중요도 (높을수록 기능적으로 중요)"
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                    except Exception as e:
                        st.error(f"오류: {str(e)}")

        # 서열 유사도 비교
        st.markdown("---")
        st.markdown("### 서열 유사도 비교")

        col1, col2 = st.columns(2)
        with col1:
            seq_a = st.text_input("서열 A", value="ACDEFGHIKLMNPQRSTVWY", key="sim_a")
        with col2:
            seq_b = st.text_input("서열 B", value="ACDEFGHIKLMNPQRSTVWY", key="sim_b")

        if st.button("유사도 계산", key="sim_btn"):
            with st.spinner("임베딩 기반 유사도 계산 중..."):
                try:
                    embedder = get_plm_embedder(model_choice, finetuned_name)
                    similarity = embedder.compute_similarity(seq_a.upper(), seq_b.upper())
                    st.metric("코사인 유사도", f"{similarity:.4f}")

                    if similarity > 0.9:
                        st.success("🟢 매우 높은 유사도 - 기능적으로 유사할 가능성 높음")
                    elif similarity > 0.7:
                        st.info("🟡 중간 유사도")
                    else:
                        st.warning("🔴 낮은 유사도 - 기능적으로 다를 가능성")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

    # ============================================================
    # 탭 2: Zero-shot 변이 예측
    # ============================================================
    with tab2:
        st.markdown("## Zero-shot 변이 효과 예측")
        st.markdown(
            "ESM-2가 학습한 진화적 정보를 활용하여, **추가 학습 없이** "
            "변이가 단백질 기능에 미치는 영향을 예측합니다."
        )

        default_wt = transferred_seq if transferred_seq else "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHISDVHCEASKSYLNN"
        wt_sequence = st.text_area(
            "야생형(Wild-type) 서열",
            value=default_wt,
            height=80,
            key="wt_seq",
            help="2번/3번 페이지에서 전송된 서열이 자동 입력됩니다."
        )

        mutations_input = st.text_input(
            "변이 목록 (쉼표로 구분)",
            value="A5G, L10V, K15R, Q20A, S30T",
            help="형식: {원래AA}{위치}{변이AA}, 예: A5G = 5번 위치 Ala→Gly"
        )

        if st.button("🔬 변이 효과 예측", key="mutation_btn"):
            mutations = [m.strip() for m in mutations_input.split(",")]
            clean_wt = "".join(c for c in wt_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")

            with st.spinner("Zero-shot scoring 중..."):
                try:
                    embedder = get_plm_embedder(model_choice, finetuned_name)
                    results = embedder.zero_shot_score(clean_wt, mutations)

                    # 결과 테이블
                    rows = []
                    for mut, data in results.items():
                        if "error" in data:
                            rows.append({
                                "Mutation": mut, "Score": "N/A",
                                "Effect": data["error"],
                                "WT Prob": "N/A", "MT Prob": "N/A"
                            })
                        else:
                            rows.append({
                                "Mutation": mut,
                                "Score": data["score"],
                                "Effect": data["effect"],
                                "WT Prob": data["wt_prob"],
                                "MT Prob": data["mt_prob"]
                            })

                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)

                    # 스코어 시각화
                    valid = [r for r in rows if r["Score"] != "N/A"]
                    if valid:
                        fig = px.bar(
                            pd.DataFrame(valid),
                            x="Mutation", y="Score",
                            color="Effect",
                            color_discrete_map={
                                "beneficial": "#2ecc71",
                                "neutral": "#f39c12",
                                "deleterious": "#e74c3c"
                            },
                            title="변이별 Fitness Score (양수=유리, 음수=유해)"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig, use_container_width=True)

                    st.info(
                        "**해석 가이드**\n"
                        "- Score > 0.5: 🟢 유리한 변이 (기능 향상 예상)\n"
                        "- -0.5 < Score < 0.5: 🟡 중립 변이\n"
                        "- Score < -0.5: 🔴 유해한 변이 (기능 저하 예상)"
                    )

                except Exception as e:
                    st.error(f"오류: {str(e)}")

    # ============================================================
    # 탭 3: 딥러닝 서열 생성
    # ============================================================
    with tab3:
        st.markdown("## 딥러닝 기반 서열 생성")

        method = st.radio(
            "생성 방법 선택",
            ["ESM-2 Masked Generation", "ESM-2 Iterative Refinement",
             "ProtGPT2 (사전학습 LM)", "VAE (학습 필요)"],
            index=0
        )

        if method == "ESM-2 Masked Generation":
            st.markdown(
                "서열의 특정 위치를 마스크하고, ESM-2가 진화적으로 타당한 "
                "아미노산을 제안합니다."
            )

            template_seq = st.text_input(
                "템플릿 서열", value="ACDEFGHIKLMNPQRSTVWY", key="mask_template"
            )
            positions_input = st.text_input(
                "마스크 위치 (쉼표 구분, 비우면 전체)",
                value="5, 10, 15",
                help="1-based 위치 번호"
            )
            n_variants = st.slider("위치당 변이체 수", 1, 10, 3)

            if st.button("🧬 변이체 생성", key="mask_gen_btn"):
                positions = None
                if positions_input.strip():
                    positions = [int(p.strip()) for p in positions_input.split(",")]

                with st.spinner("ESM-2 마스크 생성 중..."):
                    try:
                        gen_manager = get_deep_generator()
                        gen_manager._plm_embedder = get_plm_embedder(model_choice, finetuned_name)
                        results = gen_manager.generate(
                            "esm_masked",
                            sequence=template_seq.upper(),
                            positions=positions,
                            n_per_position=n_variants
                        )

                        if results:
                            df = pd.DataFrame(results)
                            display_cols = ["mutation", "position", "wt_aa", "mt_aa", "probability"]
                            st.dataframe(df[display_cols], use_container_width=True)

                            st.metric("생성된 변이체 수", len(results))
                        else:
                            st.warning("생성된 변이체가 없습니다.")

                    except Exception as e:
                        st.error(f"오류: {str(e)}")

        elif method == "ESM-2 Iterative Refinement":
            st.markdown(
                "반복적 마스크 채우기로 서열을 점진적으로 최적화합니다."
            )

            start_seq = st.text_input("시작 서열", value="ACDEFGHIKLMNPQRSTVWY", key="refine_start")
            n_iters = st.slider("반복 횟수", 1, 20, 5)
            n_muts = st.slider("반복당 변이 수", 1, 5, 2)

            if st.button("🔄 반복 최적화 시작", key="refine_btn"):
                with st.spinner(f"ESM-2 반복 최적화 ({n_iters}회)..."):
                    try:
                        gen_manager = get_deep_generator()
                        gen_manager._plm_embedder = get_plm_embedder(model_choice, finetuned_name)
                        trajectory = gen_manager.generate(
                            "esm_refinement",
                            sequence=start_seq.upper(),
                            n_iterations=n_iters,
                            n_mutations_per_iter=n_muts
                        )

                        st.markdown("### 최적화 궤적")
                        for step in trajectory:
                            muts = ", ".join(step["mutations"]) if step["mutations"] else "(시작)"
                            st.write(
                                f"**Iter {step['iteration']}**: "
                                f"`{step['sequence'][:50]}{'...' if len(step['sequence']) > 50 else ''}` "
                                f"| 변이: {muts}"
                            )

                        st.success(f"✅ 최종 서열: `{trajectory[-1]['sequence']}`")

                    except Exception as e:
                        st.error(f"오류: {str(e)}")

        elif method == "ProtGPT2 (사전학습 LM)":
            if not TRANSFORMERS_OK:
                st.error("transformers 패키지가 설치되지 않았습니다.")
            else:
                st.markdown("사전학습된 ProtGPT2로 새로운 단백질 서열을 생성합니다.")

                prompt = st.text_input("시작 프롬프트 (선택)", value="", key="gpt2_prompt")
                n_seqs = st.slider("생성 서열 수", 1, 20, 5)
                temperature = st.slider("Temperature", 0.5, 2.0, 1.0, 0.1)
                max_len = st.slider("최대 길이", 10, 200, 50)

                if st.button("🤖 ProtGPT2 생성", key="gpt2_btn"):
                    with st.spinner("ProtGPT2 로딩 및 생성 중... (첫 실행 시 모델 다운로드)"):
                        try:
                            gen_manager = get_deep_generator()
                            results = gen_manager.generate(
                                "protgpt2",
                                prompt=prompt,
                                n=n_seqs,
                                temperature=temperature,
                                max_length=max_len
                            )

                            if results:
                                df = pd.DataFrame(results)
                                st.dataframe(df[["sequence", "length"]], use_container_width=True)
                            else:
                                st.warning("생성 결과가 없습니다.")

                        except Exception as e:
                            st.error(f"오류: {str(e)}")

        elif method == "VAE (학습 필요)":
            st.markdown(
                "VAE를 자체 데이터로 학습시킨 후, "
                "잠재 공간에서 새로운 서열을 생성합니다."
            )

            st.info(
                "💡 VAE는 학습 데이터가 필요합니다.\n\n"
                "1. 서열 데이터 입력 (FASTA 또는 직접 입력)\n"
                "2. '학습 시작' 클릭\n"
                "3. 학습 완료 후 생성 가능"
            )

            training_seqs = st.text_area(
                "학습용 서열 (줄바꿈으로 구분)",
                value="ACDEFGHIKL\nMNPQRSTVWY\nACDEFGHIKL\nKLMNPQRSTV",
                height=150,
                key="vae_training"
            )

            col1, col2 = st.columns(2)
            with col1:
                vae_epochs = st.number_input("학습 에폭", 50, 500, 100)
            with col2:
                vae_latent = st.number_input("잠재 차원", 8, 128, 32)

            if st.button("📚 VAE 학습 시작", key="vae_train_btn"):
                seqs = [s.strip().upper() for s in training_seqs.strip().split("\n") if s.strip()]
                valid_seqs = [
                    "".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY")
                    for s in seqs
                ]
                valid_seqs = [s for s in valid_seqs if len(s) >= 3]

                if len(valid_seqs) < 10:
                    st.warning("최소 10개 이상의 서열이 필요합니다.")
                else:
                    with st.spinner(f"VAE 학습 중 ({vae_epochs} epochs)..."):
                        try:
                            from deep_generator import PeptideVAE
                            vae = PeptideVAE(latent_dim=vae_latent)
                            vae.train(valid_seqs, epochs=vae_epochs)
                            st.session_state["vae_trained"] = True
                            st.session_state["vae_model"] = vae
                            st.success(f"✅ VAE 학습 완료! ({len(valid_seqs)}개 서열)")
                        except Exception as e:
                            st.error(f"학습 오류: {str(e)}")

            if st.session_state.get("vae_trained"):
                n_gen = st.slider("생성 서열 수", 5, 50, 10, key="vae_n")
                temp = st.slider("Temperature", 0.5, 2.0, 1.0, 0.1, key="vae_temp")

                if st.button("🎲 VAE 서열 생성", key="vae_gen_btn"):
                    vae = st.session_state["vae_model"]
                    results = vae.generate(n=n_gen, temperature=temp)
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)

    # ============================================================
    # 탭 4: DB + ESM-2 기반 활성 예측
    # ============================================================
    with tab4:
        st.markdown("## DB + ESM-2 기반 생리활성 예측")
        st.markdown(
            "**4,162개 생리활성 펩타이드 DB** (BIOPEP-UWM 기반, 25가지 활성 유형)와 "
            "**ESM-2 임베딩 유사도**를 결합하여 입력 서열의 생리활성을 예측합니다.\n\n"
            "3번 페이지의 Markov+ESM-2+DB 파이프라인과 동일한 DB를 사용하되, "
            "여기서는 **개별 서열 단위**로 더 정밀한 분석을 제공합니다."
        )

        default_pred = transferred_seq if transferred_seq else "ACDEFGHIKLMNPQRSTVWY"
        pred_sequence = st.text_area(
            "예측할 서열",
            value=default_pred,
            height=80,
            key="pred_seq",
            help="2번/3번 페이지에서 전송된 서열이 자동 입력됩니다."
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            min_motif_len = st.slider("최소 모티프 길이", 2, 5, 3, key="ai_min_motif",
                                      help="짧은 디펩타이드 노이즈 제거용")
        with col_opt2:
            use_esm_similarity = st.checkbox("ESM-2 임베딩 유사도 포함", value=True,
                                             help="DB 매칭 + ESM-2 코사인 유사도 결합")

        if st.button("📊 활성 예측", key="predict_btn"):
            clean_pred = "".join(c for c in pred_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")

            if len(clean_pred) < 3:
                st.error("서열이 너무 짧습니다 (최소 3잔기)")
            else:
                with st.spinner("DB 매칭 + ESM-2 분석 중..."):
                    try:
                        # 1) DB 모티프 매칭
                        from bioactive_predictor import BioactiveMotifFinder
                        motif_finder = BioactiveMotifFinder()
                        motifs_found = motif_finder.find_motifs_in_sequence(
                            clean_pred, min_motif_length=min_motif_len
                        )

                        # 2) 활성별 점수 집계
                        activity_hits = {}
                        matched_details = []
                        for m in motifs_found:
                            acts = m.get('all_activities', [m['activity']])
                            for act in acts:
                                if act and act != 'unknown':
                                    activity_hits[act] = activity_hits.get(act, 0) + 1
                            matched_details.append({
                                "모티프": m['motif'],
                                "위치": m['position'],
                                "활성": ", ".join(m.get('all_activities', [m['activity']])),
                                "설명": m.get('description', ''),
                                "IC50": m.get('IC50', '-')
                            })

                        # 3) ESM-2 Fitness Score
                        esm_fitness = None
                        if use_esm_similarity:
                            embedder = get_plm_embedder(model_choice, finetuned_name)
                            fitness_results = embedder.get_batch_fitness_scores([clean_pred])
                            if fitness_results:
                                esm_fitness = fitness_results[0]

                        # 4) 정규화 및 레이더 차트
                        if activity_hits:
                            max_hits = max(activity_hits.values())
                            activity_scores = {
                                act: round(count / max_hits, 4)
                                for act, count in sorted(
                                    activity_hits.items(),
                                    key=lambda x: x[1], reverse=True
                                )
                            }

                            # Top 12 활성만 레이더에 표시
                            top_activities = list(activity_scores.keys())[:12]
                            top_scores = [activity_scores[a] for a in top_activities]

                            fig = go.Figure()
                            fig.add_trace(go.Scatterpolar(
                                r=top_scores + [top_scores[0]],
                                theta=top_activities + [top_activities[0]],
                                fill='toself',
                                name='DB Match Score',
                                fillcolor='rgba(31, 119, 180, 0.3)',
                                line=dict(color='#1f77b4')
                            ))
                            fig.update_layout(
                                polar=dict(radialaxis=dict(range=[0, 1])),
                                title="생리활성 예측 프로필 (DB 매칭 기반)",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # 메트릭 카드
                            mc1, mc2, mc3 = st.columns(3)
                            with mc1:
                                st.metric("DB 매칭 모티프", f"{len(motifs_found)}개")
                            with mc2:
                                st.metric("감지된 활성 유형", f"{len(activity_hits)}개")
                            with mc3:
                                if esm_fitness is not None:
                                    st.metric("ESM-2 Fitness", f"{esm_fitness:.4f}")
                                else:
                                    st.metric("ESM-2 Fitness", "N/A")

                            # 활성 테이블
                            st.markdown("### 활성별 상세 점수")
                            act_rows = []
                            for act, score in activity_scores.items():
                                level = "🟢 높음" if score >= 0.7 else ("🟡 중간" if score >= 0.4 else "🔵 낮음")
                                act_rows.append({
                                    "Activity": act,
                                    "Hit Count": activity_hits[act],
                                    "Score": score,
                                    "Level": level
                                })
                            st.dataframe(pd.DataFrame(act_rows), use_container_width=True, hide_index=True)

                            # 매칭된 모티프 상세
                            if matched_details:
                                st.markdown("### 매칭된 모티프 상세")
                                st.dataframe(pd.DataFrame(matched_details), use_container_width=True, hide_index=True)

                            # Top 3 분석
                            st.markdown("### Top 3 활성 분석")
                            top3 = list(activity_scores.items())[:3]
                            for i, (act, score) in enumerate(top3):
                                relevant = [m for m in motifs_found
                                           if act in m.get('all_activities', [m['activity']])]
                                seqs = list(set(m['motif'] for m in relevant))
                                st.markdown(
                                    f"**{i+1}. {act}** (Score: {score:.3f}) — "
                                    f"매칭 모티프 {len(relevant)}개: "
                                    f"`{'`, `'.join(seqs[:5])}`"
                                    f"{'...' if len(seqs) > 5 else ''}"
                                )

                        else:
                            st.warning(
                                f"입력 서열 `{clean_pred}`에서 DB 매칭 모티프가 발견되지 않았습니다.\n\n"
                                "서열이 너무 짧거나 DB에 없는 유형일 수 있습니다."
                            )
                            if esm_fitness is not None:
                                st.info(f"ESM-2 Fitness Score: **{esm_fitness:.4f}** "
                                       f"(1.0에 가까울수록 자연에서 발생 가능성 높음)")

                    except Exception as e:
                        st.error(f"오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


    # ============================================================
    # 탭 5: 배치 분석 (2번/3번 페이지 연계)
    # ============================================================
    with tab5:
        st.markdown("## 📦 배치 서열 분석")
        st.markdown(
            "2번(서열 생성) 또는 3번(모티프 검색) 페이지에서 전송된 **여러 서열을 한 번에** 분석합니다."
        )

        # 배치 서열 소스 확인
        batch_seqs = transferred_batch or []
        batch_source = transfer_source or "없음"

        if not batch_seqs:
            st.info(
                "📌 **배치 서열을 불러오는 방법:**\n\n"
                "1. **2번 페이지** → 서열 생성 → '상위 N개 서열 일괄 전송' 클릭\n"
                "2. **3번 페이지** → Activity Profile → 'Hit 서열 → AI 배치 분석' 클릭\n"
                "3. 이 페이지로 돌아오면 자동 로드됩니다.\n\n"
                "또는 아래에 직접 입력하세요."
            )

        # 직접 입력도 가능
        manual_batch = st.text_area(
            "서열 직접 입력 (줄바꿈으로 구분)",
            value="\n".join(batch_seqs) if batch_seqs else "",
            height=200,
            key="batch_manual",
            help="2번/3번 페이지에서 전송된 서열이 자동 입력됩니다. 직접 수정도 가능합니다."
        )

        if batch_seqs:
            st.success(f"📨 전송 소스: {batch_source} | {len(batch_seqs)}개 서열 로드됨")

        # 분석 옵션
        st.markdown("### 분석 옵션")
        col1, col2 = st.columns(2)
        with col1:
            do_embedding = st.checkbox("ESM-2 임베딩 유사도 분석", value=True)
            do_fitness = st.checkbox("Zero-shot Fitness 스코어링", value=True)
        with col2:
            do_activity = st.checkbox("생리활성 예측", value=True)
            do_clustering = st.checkbox("서열 클러스터링", value=True)

        if st.button("🚀 배치 분석 실행", key="batch_run", type="primary"):
            # 서열 파싱
            seqs = [
                "".join(c for c in line.strip().upper() if c in "ACDEFGHIKLMNPQRSTVWY")
                for line in manual_batch.strip().split("\n")
                if line.strip()
            ]
            seqs = [s for s in seqs if len(s) >= 3]

            if len(seqs) < 2:
                st.error("최소 2개 이상의 유효한 서열이 필요합니다.")
            else:
                embedder = get_plm_embedder(model_choice, finetuned_name)
                progress = st.progress(0, text="분석 시작...")

                # ---- 임베딩 추출 ----
                all_embeddings = []
                for i, seq in enumerate(seqs):
                    progress.progress(
                        (i + 1) / len(seqs),
                        text=f"임베딩 추출 중... ({i+1}/{len(seqs)})"
                    )
                    try:
                        emb = embedder.get_sequence_embedding(seq)
                        all_embeddings.append(emb)
                    except Exception:
                        all_embeddings.append(None)

                progress.progress(1.0, text="분석 완료!")

                valid_indices = [i for i, e in enumerate(all_embeddings) if e is not None]
                valid_seqs = [seqs[i] for i in valid_indices]
                valid_embs = [all_embeddings[i] for i in valid_indices]

                st.success(f"✅ {len(valid_seqs)}/{len(seqs)}개 서열 분석 완료")

                # ---- Zero-shot Fitness ----
                if do_fitness and valid_seqs:
                    st.markdown("### 🔬 Zero-shot Fitness 스코어")
                    fitness_scores = []
                    for seq in valid_seqs:
                        try:
                            score = embedder.get_sequence_log_likelihood(seq)
                            fitness_scores.append(score)
                        except Exception:
                            fitness_scores.append(0.0)

                    fitness_df = pd.DataFrame({
                        "서열": [s[:30] + "..." if len(s) > 30 else s for s in valid_seqs],
                        "전체 서열": valid_seqs,
                        "길이": [len(s) for s in valid_seqs],
                        "Fitness Score": fitness_scores
                    }).sort_values("Fitness Score", ascending=False)

                    st.dataframe(
                        fitness_df[["서열", "길이", "Fitness Score"]],
                        use_container_width=True, hide_index=True
                    )

                    fig = px.bar(
                        fitness_df, x="서열", y="Fitness Score",
                        color="Fitness Score",
                        color_continuous_scale="RdYlGn",
                        title="서열별 Fitness Score (높을수록 진화적으로 타당)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Top 추천
                    best = fitness_df.iloc[0]
                    st.info(f"🏆 **최고 Fitness 서열**: `{best['전체 서열']}` (Score: {best['Fitness Score']:.4f})")

                # ---- 유사도 행렬 ----
                if do_embedding and len(valid_embs) >= 2:
                    st.markdown("### 🧬 임베딩 기반 유사도 행렬")
                    import numpy as np_batch
                    emb_matrix = np_batch.array(valid_embs)
                    norms = np_batch.linalg.norm(emb_matrix, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    normalized = emb_matrix / norms
                    sim_matrix = normalized @ normalized.T

                    labels = [s[:15] + "..." if len(s) > 15 else s for s in valid_seqs]
                    fig = go.Figure(data=go.Heatmap(
                        z=sim_matrix, x=labels, y=labels,
                        colorscale="Blues", zmin=0, zmax=1
                    ))
                    fig.update_layout(title="서열 간 코사인 유사도", height=500)
                    st.plotly_chart(fig, use_container_width=True)

                # ---- 클러스터링 ----
                if do_clustering and len(valid_embs) >= 3:
                    st.markdown("### 📊 서열 클러스터링 (PCA + K-means)")
                    from sklearn.decomposition import PCA
                    from sklearn.cluster import KMeans

                    emb_matrix = np.array(valid_embs)
                    n_clusters = min(3, len(valid_embs))

                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(emb_matrix)

                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(emb_matrix)

                    cluster_df = pd.DataFrame({
                        "PC1": coords[:, 0], "PC2": coords[:, 1],
                        "Cluster": [f"Cluster {c}" for c in clusters],
                        "서열": [s[:20] + "..." if len(s) > 20 else s for s in valid_seqs],
                        "전체 서열": valid_seqs
                    })

                    fig = px.scatter(
                        cluster_df, x="PC1", y="PC2",
                        color="Cluster", hover_data=["서열"],
                        title=f"ESM-2 임베딩 기반 서열 클러스터링 ({n_clusters} clusters)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ---- 활성 예측 (DB 기반) ----
                if do_activity and valid_seqs:
                    st.markdown("### 💊 DB 기반 배치 생리활성 예측")
                    st.caption("4,162개 생리활성 펩타이드 DB 매칭 기반")

                    from bioactive_predictor import BioactiveMotifFinder
                    motif_finder = BioactiveMotifFinder()

                    activity_rows = []
                    all_activities_set = set()
                    for seq in valid_seqs:
                        motifs = motif_finder.find_motifs_in_sequence(seq, min_motif_length=3)
                        act_counts = {}
                        for m in motifs:
                            for act in m.get('all_activities', [m['activity']]):
                                if act and act != 'unknown':
                                    act_counts[act] = act_counts.get(act, 0) + 1
                                    all_activities_set.add(act)
                        activity_rows.append({
                            "서열": seq[:25] + "..." if len(seq) > 25 else seq,
                            "전체 서열": seq,
                            "모티프 수": len(motifs),
                            "_act_counts": act_counts
                        })

                    # 활성별 점수 정규화
                    top_acts = sorted(all_activities_set,
                                     key=lambda a: sum(r["_act_counts"].get(a, 0) for r in activity_rows),
                                     reverse=True)[:10]

                    for row in activity_rows:
                        for act in top_acts:
                            row[act] = row["_act_counts"].get(act, 0)

                    act_df = pd.DataFrame(activity_rows)
                    display_cols = ["서열", "모티프 수"] + top_acts
                    st.dataframe(act_df[display_cols], use_container_width=True, hide_index=True)

                    # 활성별 Top 서열
                    st.markdown("#### 🏆 활성별 최고 후보")
                    for act in top_acts[:5]:
                        if act_df[act].max() > 0:
                            best_idx = act_df[act].idxmax()
                            st.write(
                                f"**{act}**: `{act_df.loc[best_idx, '서열']}` "
                                f"(매칭 {act_df.loc[best_idx, act]}회)"
                            )


if __name__ == "__main__":
    main()
