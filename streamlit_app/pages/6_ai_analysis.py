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
def get_plm_embedder(model_name="esm2_t6_8M"):
    from plm_embedder import PLMEmbedder
    embedder = PLMEmbedder(model_name=model_name)
    embedder.load_model()
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

    # ---- 탭 구성 ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧬 서열 임베딩", "🔬 변이 예측", "🧪 DL 서열 생성", "📊 활성 예측"
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

        col1, col2 = st.columns([2, 1])
        with col1:
            input_sequence = st.text_area(
                "아미노산 서열 입력",
                value="ACDEFGHIKLMNPQRSTVWY",
                height=100,
                help="표준 20종 아미노산 문자만 입력"
            )
        with col2:
            st.markdown("**입력 정보**")
            clean_seq = "".join(c for c in input_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            st.write(f"서열 길이: {len(clean_seq)}")
            st.write(f"선택 모델: {model_choice}")

        if st.button("🚀 임베딩 추출", key="embed_btn"):
            if len(clean_seq) < 3:
                st.error("서열이 너무 짧습니다 (최소 3잔기)")
            else:
                with st.spinner(f"ESM-2 ({model_choice}) 모델 로딩 및 임베딩 추출 중..."):
                    try:
                        embedder = get_plm_embedder(model_choice)
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
                    embedder = get_plm_embedder(model_choice)
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

        wt_sequence = st.text_area(
            "야생형(Wild-type) 서열",
            value="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHISDVHCEASKSYLNN",
            height=80,
            key="wt_seq"
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
                    embedder = get_plm_embedder(model_choice)
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
                        gen_manager._plm_embedder = get_plm_embedder(model_choice)
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
                        gen_manager._plm_embedder = get_plm_embedder(model_choice)
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
    # 탭 4: ML 기반 활성 예측
    # ============================================================
    with tab4:
        st.markdown("## ML 기반 활성 예측")
        st.markdown(
            "ESM-2 임베딩을 기반으로 6가지 생리활성을 예측합니다.\n\n"
            "기존 규칙 기반 예측(3번 페이지)보다 정확한 딥러닝 예측입니다."
        )

        pred_sequence = st.text_area(
            "예측할 서열",
            value="ACDEFGHIKLMNPQRSTVWY",
            height=80,
            key="pred_seq"
        )

        use_uncertainty = st.checkbox("불확실성 추정 포함 (MC Dropout)", value=True)

        if st.button("📊 활성 예측", key="predict_btn"):
            clean_pred = "".join(c for c in pred_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")

            with st.spinner("ESM-2 임베딩 추출 + 활성 예측 중..."):
                try:
                    embedder = get_plm_embedder(model_choice)
                    embedding = embedder.get_sequence_embedding(clean_pred)

                    from fitness_predictor import FitnessPredictor
                    predictor = FitnessPredictor(
                        embedding_dim=embedder._model_info.get("dim", 320)
                    )

                    if use_uncertainty:
                        results = predictor.predict_with_uncertainty(embedding)
                    else:
                        results = predictor.predict(embedding)

                    # 결과 시각화
                    activities = list(results.keys())
                    scores = [results[a]["score"] for a in activities]
                    confidences = [results[a]["confidence"] for a in activities]

                    # 레이더 차트
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=scores + [scores[0]],
                        theta=activities + [activities[0]],
                        fill='toself',
                        name='Prediction',
                        fillcolor='rgba(31, 119, 180, 0.3)',
                        line=dict(color='#1f77b4')
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 1])),
                        title="생리활성 예측 프로필",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 상세 테이블
                    rows = []
                    for activity, data in results.items():
                        row = {
                            "Activity": activity,
                            "Score": data["score"],
                            "Confidence": data["confidence"]
                        }
                        if "uncertainty" in data:
                            row["Uncertainty"] = data["uncertainty"]
                            row["Range"] = f"{data['range'][0]:.3f} ~ {data['range'][1]:.3f}"
                        if "note" in data:
                            row["Note"] = data["note"]
                        rows.append(row)

                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    if any("note" in results[a] for a in activities):
                        st.warning(
                            "⚠️ 학습된 모델이 없어 규칙 기반 fallback을 사용했습니다.\n"
                            "DMS 데이터로 모델을 학습시키면 정확도가 크게 향상됩니다."
                        )

                except Exception as e:
                    st.error(f"오류: {str(e)}")


if __name__ == "__main__":
    main()
