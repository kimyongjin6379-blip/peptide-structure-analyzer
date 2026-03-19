"""
Page 7: ESM-2 펩톤 특화 Fine-tuning

범용 ESM-2를 펩톤 유래 펩타이드 데이터로 추가 학습시켜
예측 정확도를 향상시키는 페이지.
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

st.set_page_config(page_title="Fine-tuning", page_icon="🎯", layout="wide")


def check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


TORCH_OK = check_torch()


@st.cache_resource
def get_data_loader():
    from data_loader import CompositionLoader
    loader = CompositionLoader()
    loader.load_data()
    return loader


def main():
    st.markdown("# 🎯 ESM-2 펩톤 특화 Fine-tuning")
    st.markdown(
        "범용 ESM-2를 **우리 펩톤 데이터에 특화**시켜 "
        "펩타이드 예측 정확도를 높입니다."
    )

    if not TORCH_OK:
        st.error("이 페이지는 **Railway 배포 환경**에서만 사용 가능합니다.")
        return

    # 사이드바
    with st.sidebar:
        st.markdown("### 🧠 모델 설정")
        model_choice = st.selectbox(
            "베이스 모델",
            ["esm2_t6_8M", "esm2_t12_35M"],
            index=0,
            help="esm2_t6_8M: 빠르고 가벼움 (권장)\nesm2_t12_35M: 더 정확하지만 느림"
        )

        st.markdown("---")
        st.markdown("### 📊 저장된 모델")
        try:
            from plm_finetuner import PLMFineTuner
            tuner = PLMFineTuner(model_name=model_choice)
            saved_models = tuner.list_finetuned_models()
            if saved_models:
                for m in saved_models:
                    st.write(f"✅ {m['name']}")
                    st.caption(f"  {m['timestamp']} | {m['epochs']} epochs")
            else:
                st.info("저장된 모델 없음")
        except Exception:
            st.info("모델 목록 로딩 대기 중")

    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "📚 데이터 준비 & 학습",
        "🔬 Before vs After 비교",
        "📖 설명"
    ])

    # ============================================================
    # 탭 1: 데이터 준비 & 학습
    # ============================================================
    with tab1:
        st.markdown("## Step 1: 학습 데이터 구축")
        st.markdown(
            "Fine-tuning에 사용할 펩타이드 서열을 수집합니다. "
            "다양한 소스에서 자동으로 모아줍니다."
        )

        col1, col2 = st.columns(2)
        with col1:
            use_db = st.checkbox("📗 DB 모티프 서열 포함", value=True,
                                 help="bioactive_peptide_db.json의 알려진 모티프")
            use_known = st.checkbox("📘 알려진 식품 유래 펩타이드 포함", value=True,
                                    help="ACE 억제, 항산화, 항균 등 100+ 서열")
            use_markov = st.checkbox("📙 Markov 생성 서열 포함", value=True,
                                     help="각 샘플의 조성 기반 생성 서열")
        with col2:
            markov_n = st.number_input("샘플당 Markov 서열 수", 10, 500, 100)
            min_len = st.number_input("최소 서열 길이", 2, 10, 3)
            max_len = st.number_input("최대 서열 길이", 10, 100, 50)

        # 사용자 직접 입력
        custom_input = st.text_area(
            "📝 추가 서열 직접 입력 (줄바꿈으로 구분, 선택)",
            value="",
            height=100,
            help="자체 실험에서 확인된 펩타이드 서열을 추가할 수 있습니다"
        )

        if st.button("📊 데이터셋 구축", key="build_data", type="primary"):
            with st.spinner("학습 데이터 수집 중..."):
                from plm_finetuner import PeptoneTrainingDataBuilder

                builder = PeptoneTrainingDataBuilder()
                loader = get_data_loader() if use_markov else None

                custom_seqs = [s.strip() for s in custom_input.split("\n") if s.strip()] if custom_input.strip() else None

                dataset = builder.build_training_set(
                    include_db_motifs=use_db,
                    include_known_peptides=use_known,
                    include_markov_seqs=use_markov,
                    custom_sequences=custom_seqs,
                    markov_loader=loader,
                    markov_n_per_sample=markov_n,
                    min_length=min_len,
                    max_length=max_len,
                )

                st.session_state['ft_dataset'] = dataset

        # 데이터셋 표시
        if 'ft_dataset' in st.session_state:
            dataset = st.session_state['ft_dataset']
            stats = dataset['stats']

            st.success(f"✅ 학습 데이터 구축 완료: **{stats['total_unique']}개** 고유 서열")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 고유 서열", stats['total_unique'])
            with col2:
                st.metric("평균 길이", f"{stats['avg_length']} AA")
            with col3:
                st.metric("최소 길이", stats['length_distribution']['min'])
            with col4:
                st.metric("최대 길이", stats['length_distribution']['max'])

            # 소스별 통계
            st.markdown("#### 데이터 소스별 서열 수")
            source_df = pd.DataFrame([
                {"소스": "DB 모티프", "서열 수": stats['by_source']['db_motifs']},
                {"소스": "알려진 식품 펩타이드", "서열 수": stats['by_source']['known_peptides']},
                {"소스": "Markov 생성", "서열 수": stats['by_source']['markov_generated']},
                {"소스": "사용자 입력", "서열 수": stats['by_source']['custom']},
            ])
            st.dataframe(source_df, use_container_width=True, hide_index=True)

            # 길이 분포
            lengths = [len(s) for s in dataset['sequences']]
            fig = px.histogram(x=lengths, nbins=30, title="서열 길이 분포",
                               labels={"x": "서열 길이 (AA)", "y": "개수"})
            st.plotly_chart(fig, use_container_width=True)

            # ---- Step 2: Fine-tuning ----
            st.markdown("---")
            st.markdown("## Step 2: Fine-tuning 실행")

            col1, col2, col3 = st.columns(3)
            with col1:
                ft_epochs = st.number_input("학습 에폭", 3, 50, 10,
                                            help="10~20이 적당. 너무 많으면 overfitting")
            with col2:
                ft_lr = st.select_slider(
                    "학습률",
                    options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                    value=1e-5,
                    help="1e-5 권장. 너무 크면 기존 지식 손실"
                )
            with col3:
                ft_freeze = st.number_input(
                    "동결 레이어 수", 0, 5, 0,
                    help="0=전체 학습, 4=상위 2개 레이어만 학습 (overfitting 방지)"
                )

            mask_ratio = st.slider("마스킹 비율", 0.05, 0.30, 0.15,
                                   help="15%가 표준 (BERT와 동일)")

            if st.button("🚀 Fine-tuning 시작", key="start_ft", type="primary"):
                from plm_finetuner import PLMFineTuner

                finetuner = PLMFineTuner(model_name=model_choice)

                progress_bar = st.progress(0, text="모델 로딩 중...")
                status_text = st.empty()
                chart_placeholder = st.empty()

                # 프로그레스 콜백
                history_for_chart = []

                def on_progress(epoch, total, info):
                    progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
                    status_text.write(
                        f"**Epoch {epoch}** | "
                        f"Train Loss: {info['train_loss']:.4f} | "
                        f"Val Loss: {info['val_loss']:.4f} | "
                        f"Val PPL: {info['val_perplexity']:.1f} | "
                        f"경과: {info['elapsed_sec']}초"
                    )
                    history_for_chart.append(info)

                    if len(history_for_chart) > 1:
                        df = pd.DataFrame(history_for_chart)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df['epoch'], y=df['train_loss'],
                            mode='lines+markers', name='Train Loss'
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['epoch'], y=df['val_loss'],
                            mode='lines+markers', name='Val Loss'
                        ))
                        fig.update_layout(
                            title="학습 곡선 (실시간)",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=300
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)

                try:
                    result = finetuner.finetune(
                        sequences=dataset['sequences'],
                        epochs=ft_epochs,
                        lr=ft_lr,
                        mask_ratio=mask_ratio,
                        freeze_layers=ft_freeze,
                        progress_callback=on_progress,
                    )

                    progress_bar.progress(1.0, text="✅ Fine-tuning 완료!")

                    st.success(
                        f"🎉 Fine-tuning 완료!\n\n"
                        f"- 학습 서열: {result['training_sequences']}개\n"
                        f"- 검증 서열: {result['validation_sequences']}개\n"
                        f"- Best Val Perplexity: {result['best_val_perplexity']}\n"
                        f"- 소요 시간: {result['total_time_sec']}초"
                    )

                    # 모델 저장
                    save_name = st.text_input("모델 저장 이름", value="peptone_finetuned")
                    if st.button("💾 Fine-tuned 모델 저장", key="save_ft"):
                        path = finetuner.save_finetuned(name=save_name)
                        st.success(f"✅ 저장 완료: {path}")
                        st.session_state['ft_model_saved'] = True
                        st.session_state['ft_model_name'] = save_name
                        st.session_state['ft_result'] = result

                except Exception as e:
                    st.error(f"Fine-tuning 오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ============================================================
    # 탭 2: Before vs After 비교
    # ============================================================
    with tab2:
        st.markdown("## 🔬 범용 ESM-2 vs 펩톤 특화 ESM-2 비교")
        st.markdown(
            "동일한 서열에 대해 Fine-tuning **전/후 모델의 차이**를 정량적으로 비교합니다."
        )

        # 저장된 모델 확인
        try:
            from plm_finetuner import PLMFineTuner
            tuner_check = PLMFineTuner(model_name=model_choice)
            saved = tuner_check.list_finetuned_models()
        except Exception:
            saved = []

        if not saved:
            st.warning(
                "⚠️ 저장된 Fine-tuned 모델이 없습니다.\n\n"
                "**📚 데이터 준비 & 학습** 탭에서 Fine-tuning을 먼저 실행하세요."
            )
            return

        selected_model = st.selectbox(
            "비교할 Fine-tuned 모델",
            [m['name'] for m in saved]
        )

        # 비교할 서열 입력
        st.markdown("### 비교할 서열")

        compare_source = st.radio(
            "서열 소스",
            ["직접 입력", "2번 페이지에서 전송된 서열", "예시 서열"],
            horizontal=True
        )

        if compare_source == "직접 입력":
            compare_seqs_input = st.text_area(
                "비교할 서열 (줄바꿈 구분)",
                value="EETEE\nIPP\nVPP\nLKPNM\nGPP",
                height=150
            )
            compare_seqs = [s.strip().upper() for s in compare_seqs_input.split("\n") if s.strip()]
        elif compare_source == "2번 페이지에서 전송된 서열":
            batch = st.session_state.get('ai_batch_sequences', [])
            if batch:
                compare_seqs = batch[:20]
                st.info(f"{len(compare_seqs)}개 서열 로드됨")
            else:
                st.warning("전송된 서열이 없습니다. 2번 페이지에서 전송하세요.")
                compare_seqs = []
        else:
            compare_seqs = [
                "IPP", "VPP", "LKP", "EETEE", "GPP",
                "ALPMHIR", "LKPNM", "YFCLT", "PHFL", "RRWQWR"
            ]
            st.info("알려진 생리활성 펩타이드 10개로 비교")

        if compare_seqs and st.button("🔬 비교 분석 실행", key="run_compare", type="primary"):
            with st.spinner("두 모델 로딩 및 비교 분석 중... (첫 실행 시 시간 소요)"):
                try:
                    from plm_embedder import PLMEmbedder
                    from plm_finetuner import PLMFineTuner, ModelComparator

                    # 범용 모델
                    base_embedder = PLMEmbedder(model_name=model_choice)
                    base_embedder.load_model()

                    # Fine-tuned 모델
                    ft_embedder = PLMEmbedder(model_name=model_choice)
                    ft_embedder.load_model()

                    # Fine-tuned 가중치 로드
                    finetuner = PLMFineTuner(model_name=model_choice)
                    if finetuner.load_finetuned(name=selected_model.replace(f"_{model_choice}", "")):
                        ft_embedder.model = finetuner.model
                        ft_embedder.model.eval()
                    else:
                        st.error("Fine-tuned 모델 로드 실패")
                        return

                    comparator = ModelComparator(base_embedder, ft_embedder)

                    # ---- 1. Perplexity 비교 ----
                    st.markdown("### 1️⃣ Perplexity 비교 (낮을수록 좋음)")
                    st.markdown(
                        "Perplexity = 모델이 서열을 얼마나 '자연스럽다'고 판단하는지의 지표. "
                        "**Fine-tuned 모델의 PPL이 더 낮으면** 펩톤 서열을 더 잘 이해한다는 뜻입니다."
                    )

                    ppl_result = comparator.compare_perplexity(compare_seqs)

                    # 차트
                    ppl_df = pd.DataFrame(ppl_result['per_sequence'])

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='범용 ESM-2',
                        x=ppl_df['sequence'], y=ppl_df['base_perplexity'],
                        marker_color='#FF6B6B'
                    ))
                    fig.add_trace(go.Bar(
                        name='펩톤 특화 ESM-2',
                        x=ppl_df['sequence'], y=ppl_df['finetuned_perplexity'],
                        marker_color='#4ECDC4'
                    ))
                    fig.update_layout(
                        barmode='group',
                        title="서열별 Perplexity 비교 (낮을수록 좋음)",
                        yaxis_title="Perplexity",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("범용 평균 PPL", ppl_result['avg_base_ppl'])
                    with col2:
                        st.metric("펩톤 특화 평균 PPL", ppl_result['avg_ft_ppl'])
                    with col3:
                        imp = ppl_result['avg_improvement_pct']
                        st.metric("평균 개선율", f"{imp}%",
                                  delta=f"{imp}% 향상" if imp > 0 else f"{imp}% 하락")

                    st.dataframe(ppl_df, use_container_width=True, hide_index=True)

                    # ---- 2. 임베딩 비교 ----
                    st.markdown("---")
                    st.markdown("### 2️⃣ 임베딩 변화 분석")
                    st.markdown(
                        "Fine-tuning으로 **임베딩이 얼마나 변했는지** 확인합니다. "
                        "큰 변화 = 모델이 해당 서열을 다르게 해석하게 됨."
                    )

                    emb_result = comparator.compare_embeddings(compare_seqs)

                    emb_df = pd.DataFrame(emb_result['per_sequence'])
                    fig_emb = px.scatter(
                        emb_df, x="cosine_similarity", y="l2_distance",
                        text="sequence", color="shift_magnitude",
                        color_discrete_map={"large": "#e74c3c", "medium": "#f39c12", "small": "#2ecc71"},
                        title="서열별 임베딩 변화 (좌하단=큰 변화, 우상단=작은 변화)"
                    )
                    fig_emb.update_traces(textposition='top center')
                    fig_emb.update_layout(height=400)
                    st.plotly_chart(fig_emb, use_container_width=True)

                    # PCA 비교
                    st.markdown("#### PCA 시각화: 임베딩 공간 변화")
                    from sklearn.decomposition import PCA

                    all_embs = np.vstack([
                        emb_result['base_embeddings'],
                        emb_result['finetuned_embeddings']
                    ])
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(all_embs)

                    n = len(compare_seqs)
                    pca_df = pd.DataFrame({
                        "PC1": coords[:, 0],
                        "PC2": coords[:, 1],
                        "서열": compare_seqs * 2,
                        "모델": ["범용 ESM-2"] * n + ["펩톤 특화 ESM-2"] * n,
                    })

                    fig_pca = px.scatter(
                        pca_df, x="PC1", y="PC2",
                        color="모델", symbol="모델",
                        hover_data=["서열"],
                        color_discrete_map={"범용 ESM-2": "#FF6B6B", "펩톤 특화 ESM-2": "#4ECDC4"},
                        title="PCA: 범용 vs 펩톤 특화 임베딩 공간"
                    )
                    # 같은 서열 연결선
                    for i in range(n):
                        fig_pca.add_trace(go.Scatter(
                            x=[coords[i, 0], coords[n + i, 0]],
                            y=[coords[i, 1], coords[n + i, 1]],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    fig_pca.update_layout(height=500)
                    st.plotly_chart(fig_pca, use_container_width=True)

                    # ---- 3. 변이 예측 비교 ----
                    st.markdown("---")
                    st.markdown("### 3️⃣ 변이 예측 비교")

                    if len(compare_seqs) > 0:
                        test_seq = max(compare_seqs, key=len)  # 가장 긴 서열로 테스트

                        if len(test_seq) >= 5:
                            # 자동 변이 생성
                            auto_mutations = []
                            aa_list = "ACDEFGHIKLMNPQRSTVWY"
                            for pos in range(0, min(len(test_seq), 10), 2):
                                orig = test_seq[pos]
                                for mt in aa_list:
                                    if mt != orig:
                                        auto_mutations.append(f"{orig}{pos+1}{mt}")
                                        break

                            if auto_mutations:
                                st.write(f"테스트 서열: `{test_seq}`")
                                st.write(f"변이: {', '.join(auto_mutations)}")

                                mut_result = comparator.compare_mutation_scores(test_seq, auto_mutations)

                                if mut_result['comparisons']:
                                    mut_df = pd.DataFrame(mut_result['comparisons'])
                                    fig_mut = go.Figure()
                                    fig_mut.add_trace(go.Bar(
                                        name='범용 ESM-2',
                                        x=mut_df['mutation'],
                                        y=mut_df['base_score'],
                                        marker_color='#FF6B6B'
                                    ))
                                    fig_mut.add_trace(go.Bar(
                                        name='펩톤 특화 ESM-2',
                                        x=mut_df['mutation'],
                                        y=mut_df['finetuned_score'],
                                        marker_color='#4ECDC4'
                                    ))
                                    fig_mut.update_layout(
                                        barmode='group',
                                        title="변이별 Fitness Score 비교",
                                        yaxis_title="Score"
                                    )
                                    fig_mut.add_hline(y=0, line_dash="dash", line_color="gray")
                                    st.plotly_chart(fig_mut, use_container_width=True)

                                    st.metric(
                                        "효과 판정이 바뀐 변이 수",
                                        f"{mut_result['n_effect_changes']}/{len(mut_result['comparisons'])}"
                                    )

                    # ---- 종합 요약 ----
                    st.markdown("---")
                    st.markdown("### 📋 종합 비교 요약")

                    summary_data = {
                        "항목": [
                            "평균 Perplexity",
                            "Perplexity 개선율",
                            "평균 임베딩 코사인 유사도",
                            "평균 임베딩 L2 거리",
                        ],
                        "범용 ESM-2": [
                            ppl_result['avg_base_ppl'],
                            "-",
                            "1.0 (자기 자신)",
                            "0.0",
                        ],
                        "펩톤 특화 ESM-2": [
                            ppl_result['avg_ft_ppl'],
                            f"{ppl_result['avg_improvement_pct']}%",
                            emb_result['avg_cosine_similarity'],
                            emb_result['avg_l2_distance'],
                        ],
                        "해석": [
                            "낮을수록 서열 이해도 높음",
                            "양수면 개선됨",
                            "1에 가까우면 변화 적음",
                            "클수록 임베딩이 많이 바뀜",
                        ]
                    }
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                    if ppl_result['avg_improvement_pct'] > 5:
                        st.success(
                            f"✅ **Fine-tuning 효과 확인!** "
                            f"Perplexity가 평균 {ppl_result['avg_improvement_pct']}% 감소했습니다. "
                            f"펩톤 유래 펩타이드를 더 잘 이해하게 되었습니다."
                        )
                    elif ppl_result['avg_improvement_pct'] > 0:
                        st.info(
                            f"📊 약간의 개선이 있습니다 ({ppl_result['avg_improvement_pct']}%). "
                            f"더 많은 학습 데이터나 에폭을 추가해보세요."
                        )
                    else:
                        st.warning(
                            "⚠️ 개선이 미미합니다. 학습률을 낮추거나 데이터를 늘려보세요."
                        )

                except Exception as e:
                    st.error(f"비교 분석 오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ============================================================
    # 탭 3: 설명
    # ============================================================
    with tab3:
        st.markdown("## 📖 Fine-tuning이란?")

        st.markdown("""
        ### 범용 ESM-2의 한계

        ESM-2는 **2억 5천만 개의 단백질 서열**로 학습된 범용 모델입니다.
        자연계의 거의 모든 단백질을 학습했지만, **펩톤 유래 펩타이드**에 특화되어 있지 않습니다.

        ```
        ESM-2가 본 데이터:         우리가 다루는 데이터:
        ┌──────────────────┐      ┌──────────────────┐
        │ 전체 단백질 2.5억개 │      │ 펩톤 펩타이드     │
        │ ┌──────────────┐ │      │ - 식품 유래       │
        │ │ 효소          │ │      │ - 짧은 서열 (3~20)│
        │ │ 구조 단백질    │ │      │ - 특정 활성 보유  │
        │ │ 막 단백질      │ │      │ - Glu/Pro 풍부   │
        │ │ ...           │ │      └──────────────────┘
        │ │ 식품 펩타이드  │◀── 아주 작은 비중!
        │ └──────────────┘ │
        └──────────────────┘
        ```

        ### Fine-tuning이 하는 일

        ESM-2의 확률 분포를 **펩톤 펩타이드 쪽으로 이동**시킵니다.

        ```
        범용 ESM-2:
          "EETEE" → Perplexity 12.5 (별로 자연스럽지 않다고 판단)
          "IPP"   → Perplexity 8.3

        펩톤 특화 ESM-2 (Fine-tuned):
          "EETEE" → Perplexity 5.2 (✅ 펩톤에선 자연스러움!)
          "IPP"   → Perplexity 3.1 (✅ 더 확신)
        ```

        ### 구체적으로 뭐가 달라지나?

        | 기능 | 범용 ESM-2 | 펩톤 특화 ESM-2 |
        |---|---|---|
        | **임베딩** | 일반적인 단백질 표현 | 펩톤 펩타이드에 최적화된 표현 |
        | **변이 예측** | 자연계 전체 기준 판단 | 펩톤 맥락에서 판단 |
        | **서열 생성** | 범용 단백질 생성 | 펩톤에서 나올법한 서열 생성 |
        | **활성 예측** | 범용 피처 사용 | 펩톤 특화 피처 → 더 정확 |
        | **유사도** | 일반 기능 유사도 | 펩톤 내 기능 유사도 |

        ### 주의사항

        - **학습률**: 1e-5 권장. 너무 크면 기존 지식 손실 (catastrophic forgetting)
        - **에폭 수**: 10~20이 적당. 과도하면 overfitting
        - **데이터 수**: 최소 50개, 200개 이상 권장
        - **레이어 동결**: overfitting 걱정되면 하위 4개 레이어 동결 추천
        """)


if __name__ == "__main__":
    main()
