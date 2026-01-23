"""
2D 시각화 모듈
2D visualization for peptide analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from .utils import AMINO_ACIDS, calculate_property_ratios
    from .data_loader import CompositionLoader
except ImportError:
    from utils import AMINO_ACIDS, calculate_property_ratios
    from data_loader import CompositionLoader


class CompositionVisualizer:
    """
    아미노산 조성 시각화
    """

    @staticmethod
    def plot_composition_bar(composition: Dict[str, float],
                            title: str = "Amino Acid Composition") -> go.Figure:
        """
        조성 바 차트

        Args:
            composition: {AA: percentage} 딕셔너리
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # 데이터 정렬 (퍼센트 내림차순)
        sorted_data = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        amino_acids = [aa for aa, _ in sorted_data]
        percentages = [pct for _, pct in sorted_data]

        # 아미노산 이름 가져오기
        aa_names = [AMINO_ACIDS.get(aa, {}).get('name', aa) for aa in amino_acids]

        # 색상 매핑 (소수성에 따라)
        colors = []
        for aa in amino_acids:
            props = AMINO_ACIDS.get(aa, {})
            if props.get('hydrophobic', False):
                colors.append('#FFA500')  # 주황색: 소수성
            elif props.get('positively_charged', False):
                colors.append('#4169E1')  # 파란색: 양전하
            elif props.get('negatively_charged', False):
                colors.append('#DC143C')  # 빨간색: 음전하
            elif props.get('polar', False):
                colors.append('#32CD32')  # 녹색: 극성
            else:
                colors.append('#808080')  # 회색: 기타

        fig = go.Figure(data=[
            go.Bar(
                x=amino_acids,
                y=percentages,
                marker_color=colors,
                text=[f"{pct:.1f}%" for pct in percentages],
                textposition='outside',
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Code: %{x}<br>' +
                             'Percentage: %{y:.2f}%<extra></extra>',
                customdata=[[name] for name in aa_names]
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Amino Acid",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_composition_comparison(compositions: Dict[str, Dict[str, float]],
                                   title: str = "Sample Comparison") -> go.Figure:
        """
        여러 샘플 조성 비교

        Args:
            compositions: {sample_id: {AA: percentage}} 딕셔너리
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # 모든 아미노산 수집
        all_aas = set()
        for comp in compositions.values():
            all_aas.update(comp.keys())
        all_aas = sorted(all_aas)

        # 각 샘플별 바 추가
        fig = go.Figure()

        for sample_id, composition in compositions.items():
            percentages = [composition.get(aa, 0) for aa in all_aas]
            fig.add_trace(go.Bar(
                name=sample_id,
                x=all_aas,
                y=percentages,
                text=[f"{pct:.1f}%" if pct > 0 else "" for pct in percentages],
                textposition='outside'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Amino Acid",
            yaxis_title="Percentage (%)",
            barmode='group',
            template="plotly_white",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def plot_property_radar(composition: Dict[str, float],
                           title: str = "Physicochemical Properties") -> go.Figure:
        """
        물리화학적 특성 레이더 차트

        Args:
            composition: {AA: percentage} 딕셔너리
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # 특성 계산
        properties = calculate_property_ratios(composition)

        categories = [
            'Hydrophobic',
            'Charged',
            'Aromatic',
            'Polar',
            'Tiny',
            'Small'
        ]

        values = [
            properties.get('hydrophobic_ratio', 0),
            properties.get('charged_ratio', 0),
            properties.get('aromatic_ratio', 0),
            properties.get('polar_ratio', 0),
            properties.get('tiny_ratio', 0),
            properties.get('small_ratio', 0)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Properties',
            line_color='#4169E1',
            fillcolor='rgba(65, 105, 225, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title=title,
            template="plotly_white",
            height=500
        )

        return fig


class MolecularWeightVisualizer:
    """
    분자량 분포 시각화
    """

    @staticmethod
    def plot_mw_distribution(distribution: Dict[str, float],
                            title: str = "Molecular Weight Distribution") -> go.Figure:
        """
        분자량 분포 히스토그램

        Args:
            distribution: {bin_name: percentage} 딕셔너리
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # 구간 이름 정리
        bin_labels = {
            'mw_pct_250': '<250 Da',
            'mw_pct_250_500': '250-500 Da',
            'mw_pct_500_750': '500-750 Da',
            'mw_pct_750_1000': '750-1000 Da',
            'mw_pct_1000': '>1000 Da'
        }

        labels = []
        percentages = []

        for key, value in distribution.items():
            label = bin_labels.get(key, key)
            labels.append(label)
            percentages.append(value)

        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=percentages,
                marker_color='#32CD32',
                text=[f"{pct:.1f}%" for pct in percentages],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Molecular Weight Range",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_mw_comparison(mw_data: pd.DataFrame,
                          title: str = "MW Distribution Comparison") -> go.Figure:
        """
        여러 샘플 분자량 분포 비교

        Args:
            mw_data: MW 분포 DataFrame (rows=samples, cols=bins)
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()

        for col in mw_data.columns:
            fig.add_trace(go.Box(
                y=mw_data[col],
                name=col,
                boxmean='sd'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="MW Range",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            height=500
        )

        return fig


class BioactivityVisualizer:
    """
    생리활성 시각화
    """

    @staticmethod
    def plot_activity_scores(activity_scores: Dict[str, float],
                            title: str = "Bioactivity Scores") -> go.Figure:
        """
        생리활성 점수 레이더 차트

        Args:
            activity_scores: {activity: score} 딕셔너리
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        activities = list(activity_scores.keys())
        scores = [activity_scores[act] * 100 for act in activities]  # 0-100 범위로

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=activities,
            fill='toself',
            name='Activity Scores',
            line_color='#DC143C',
            fillcolor='rgba(220, 20, 60, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title=title,
            template="plotly_white",
            height=500
        )

        return fig

    @staticmethod
    def plot_activity_comparison(samples_activities: Dict[str, Dict[str, float]],
                                title: str = "Activity Comparison") -> go.Figure:
        """
        여러 샘플 생리활성 비교

        Args:
            samples_activities: {sample_id: {activity: score}} 딕셔너리
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # 활성 목록
        all_activities = set()
        for activities in samples_activities.values():
            all_activities.update(activities.keys())
        all_activities = sorted(all_activities)

        fig = go.Figure()

        for sample_id, activities in samples_activities.items():
            scores = [activities.get(act, 0) * 100 for act in all_activities]
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=all_activities,
                fill='toself',
                name=sample_id
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=title,
            template="plotly_white",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig


class PeptideDiagram:
    """
    펩타이드 다이어그램
    """

    @staticmethod
    def plot_sequence_diagram(sequence: str,
                             title: str = "Peptide Sequence") -> go.Figure:
        """
        서열 다이어그램

        Args:
            sequence: 아미노산 서열
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # 각 아미노산 위치
        positions = list(range(1, len(sequence) + 1))
        amino_acids = list(sequence)

        # 색상 매핑
        colors = []
        for aa in amino_acids:
            props = AMINO_ACIDS.get(aa, {})
            if props.get('hydrophobic', False):
                colors.append('#FFA500')
            elif props.get('positively_charged', False):
                colors.append('#4169E1')
            elif props.get('negatively_charged', False):
                colors.append('#DC143C')
            elif props.get('polar', False):
                colors.append('#32CD32')
            else:
                colors.append('#808080')

        # 아미노산 이름
        aa_names = [AMINO_ACIDS.get(aa, {}).get('name', aa) for aa in amino_acids]

        fig = go.Figure()

        # 펩타이드 결합 선
        fig.add_trace(go.Scatter(
            x=positions,
            y=[1] * len(positions),
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # 아미노산 원
        fig.add_trace(go.Scatter(
            x=positions,
            y=[1] * len(positions),
            mode='markers+text',
            marker=dict(
                size=40,
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=amino_acids,
            textfont=dict(size=14, color='white', family='Arial Black'),
            textposition='middle center',
            hovertemplate='<b>Position %{x}</b><br>' +
                         'Code: %{text}<br>' +
                         'Name: %{customdata[0]}<extra></extra>',
            customdata=[[name] for name in aa_names],
            showlegend=False
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Position",
                showgrid=False,
                zeroline=False,
                range=[0, len(sequence) + 1]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[0.5, 1.5]
            ),
            template="plotly_white",
            height=300,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return fig

    @staticmethod
    def plot_hydrophobicity_profile(sequence: str,
                                    title: str = "Hydrophobicity Profile") -> go.Figure:
        """
        소수성 프로파일

        Args:
            sequence: 아미노산 서열
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        # Kyte-Doolittle 소수성 지수
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

        positions = list(range(1, len(sequence) + 1))
        hydrophobicity = [hydrophobicity_scale.get(aa, 0) for aa in sequence]

        fig = go.Figure()

        # 소수성 프로파일
        fig.add_trace(go.Scatter(
            x=positions,
            y=hydrophobicity,
            mode='lines+markers',
            line=dict(color='#4169E1', width=2),
            marker=dict(size=8),
            name='Hydrophobicity',
            hovertemplate='<b>Position %{x}</b><br>' +
                         'AA: ' + '<br>'.join([f"{sequence[i]}" for i in range(len(sequence))]) +
                         '<br>Hydrophobicity: %{y:.2f}<extra></extra>'
        ))

        # 0 기준선
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")

        fig.update_layout(
            title=title,
            xaxis_title="Position",
            yaxis_title="Hydrophobicity (Kyte-Doolittle)",
            template="plotly_white",
            height=400
        )

        return fig


if __name__ == '__main__':
    # 테스트
    print("=== 2D Visualizer Test ===\n")

    # 1. 조성 바 차트
    print("1. Composition bar chart")
    test_composition = {
        'A': 10.5, 'R': 8.2, 'N': 5.3, 'D': 7.1, 'C': 2.1,
        'E': 12.1, 'G': 6.5, 'H': 3.2, 'I': 4.8, 'L': 8.9
    }

    fig = CompositionVisualizer.plot_composition_bar(test_composition)
    print("   Created composition bar chart")

    # 2. 물리화학적 특성 레이더 차트
    print("\n2. Property radar chart")
    fig2 = CompositionVisualizer.plot_property_radar(test_composition)
    print("   Created property radar chart")

    # 3. MW 분포
    print("\n3. MW distribution")
    test_mw = {
        'mw_pct_250': 17.6,
        'mw_pct_250_500': 18.6,
        'mw_pct_500_750': 16.6,
        'mw_pct_750_1000': 13.2,
        'mw_pct_1000': 34.0
    }

    fig3 = MolecularWeightVisualizer.plot_mw_distribution(test_mw)
    print("   Created MW distribution chart")

    # 4. 생리활성 점수
    print("\n4. Bioactivity scores")
    test_activities = {
        'antimicrobial': 0.36,
        'antihypertensive': 0.27,
        'antioxidant': 0.13,
        'opioid': 0.23,
        'immunomodulatory': 0.50,
        'anti-inflammatory': 0.25
    }

    fig4 = BioactivityVisualizer.plot_activity_scores(test_activities)
    print("   Created bioactivity radar chart")

    # 5. 펩타이드 다이어그램
    print("\n5. Peptide sequence diagram")
    test_sequence = "ARNDCEQGH"

    fig5 = PeptideDiagram.plot_sequence_diagram(test_sequence)
    print(f"   Created sequence diagram for {test_sequence}")

    # 6. 소수성 프로파일
    print("\n6. Hydrophobicity profile")
    fig6 = PeptideDiagram.plot_hydrophobicity_profile(test_sequence)
    print(f"   Created hydrophobicity profile")

    print("\n[OK] 2D Visualizer test complete!")
    print("\nNote: Figures created but not displayed (use fig.show() in interactive mode)")
