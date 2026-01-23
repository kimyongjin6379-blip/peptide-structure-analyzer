"""
3D 구조 시각화 모듈 (py3Dmol)
3D structure visualization using py3Dmol
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Optional import for py3Dmol
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

# Optional import for IPython
try:
    from IPython.display import display, HTML
except ImportError:
    pass

try:
    from .utils import AMINO_ACIDS
except ImportError:
    from utils import AMINO_ACIDS


class Py3DmolViewer:
    """
    py3Dmol 기반 3D 구조 뷰어
    """

    def __init__(self, width: int = 800, height: int = 600):
        """
        초기화

        Args:
            width: 뷰어 너비
            height: 뷰어 높이
        """
        if not PY3DMOL_AVAILABLE:
            raise ImportError("py3Dmol is not installed. Install it with: pip install py3Dmol")

        self.width = width
        self.height = height
        self.viewer = None

    def create_viewer(self) -> py3Dmol.view:
        """
        뷰어 생성

        Returns:
            py3Dmol viewer 객체
        """
        self.viewer = py3Dmol.view(width=self.width, height=self.height)
        return self.viewer

    def load_pdb_file(self, pdb_path: Path) -> py3Dmol.view:
        """
        PDB 파일 로딩

        Args:
            pdb_path: PDB 파일 경로

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            self.create_viewer()

        with open(pdb_path, 'r') as f:
            pdb_content = f.read()

        self.viewer.addModel(pdb_content, 'pdb')
        return self.viewer

    def load_pdb_string(self, pdb_content: str) -> py3Dmol.view:
        """
        PDB 문자열 로딩

        Args:
            pdb_content: PDB 파일 내용

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            self.create_viewer()

        self.viewer.addModel(pdb_content, 'pdb')
        return self.viewer

    def set_style(self, style: str = 'cartoon', color: str = 'spectrum') -> py3Dmol.view:
        """
        스타일 설정

        Args:
            style: 스타일 ('cartoon', 'stick', 'sphere', 'line', 'cross')
            color: 색상 스킴 ('spectrum', 'residue', 'chain', 'hydrophobicity', 'charge')

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        # 색상 스킴 설정
        color_scheme = self._get_color_scheme(color)

        # 스타일 적용
        if style == 'cartoon':
            self.viewer.setStyle({'cartoon': {'color': color_scheme}})
        elif style == 'stick':
            self.viewer.setStyle({'stick': {'colorscheme': color_scheme}})
        elif style == 'sphere':
            self.viewer.setStyle({'sphere': {'colorscheme': color_scheme}})
        elif style == 'line':
            self.viewer.setStyle({'line': {'colorscheme': color_scheme}})
        elif style == 'cross':
            self.viewer.setStyle({'cross': {'colorscheme': color_scheme}})
        else:
            # 기본: cartoon
            self.viewer.setStyle({'cartoon': {'color': color_scheme}})

        return self.viewer

    def _get_color_scheme(self, color: str) -> str:
        """
        색상 스킴 변환

        Args:
            color: 색상 이름

        Returns:
            py3Dmol 색상 스킴
        """
        color_map = {
            'spectrum': 'spectrum',
            'residue': 'amino',
            'chain': 'chain',
            'hydrophobicity': 'hydrophobicity',
            'charge': 'electrostatic'
        }

        return color_map.get(color, 'spectrum')

    def add_surface(self, opacity: float = 0.7, color: str = 'white') -> py3Dmol.view:
        """
        표면 추가

        Args:
            opacity: 투명도 (0-1)
            color: 표면 색상

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        self.viewer.addSurface(py3Dmol.VDW, {'opacity': opacity, 'color': color})
        return self.viewer

    def highlight_residue(self, residue_number: int, color: str = 'red') -> py3Dmol.view:
        """
        특정 잔기 하이라이트

        Args:
            residue_number: 잔기 번호
            color: 하이라이트 색상

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        self.viewer.addStyle({'resi': residue_number}, {'sphere': {'color': color}})
        return self.viewer

    def zoom_to(self, selection: Dict = None) -> py3Dmol.view:
        """
        특정 영역으로 줌

        Args:
            selection: 선택 딕셔너리 (예: {'resi': [1, 2, 3]})

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        if selection:
            self.viewer.zoomTo(selection)
        else:
            self.viewer.zoomTo()

        return self.viewer

    def render(self) -> py3Dmol.view:
        """
        렌더링

        Returns:
            py3Dmol viewer 객체
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        self.viewer.zoomTo()
        return self.viewer

    def show(self):
        """
        뷰어 표시 (Jupyter 환경)
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        self.viewer.show()

    def get_html(self) -> str:
        """
        HTML 코드 가져오기

        Returns:
            HTML 문자열
        """
        if self.viewer is None:
            raise ValueError("Viewer not initialized. Load a structure first.")

        return self.viewer._make_html()


class StructureComparator:
    """
    여러 구조 비교
    """

    @staticmethod
    def create_side_by_side_view(pdb_contents: List[Tuple[str, str]],
                                 width: int = 400,
                                 height: int = 400) -> List[py3Dmol.view]:
        """
        나란히 비교 뷰

        Args:
            pdb_contents: [(이름, PDB 내용), ...] 리스트
            width: 각 뷰어 너비
            height: 각 뷰어 높이

        Returns:
            py3Dmol viewer 리스트
        """
        viewers = []

        for name, pdb_content in pdb_contents:
            viewer = py3Dmol.view(width=width, height=height)
            viewer.addModel(pdb_content, 'pdb')
            viewer.setStyle({'cartoon': {'color': 'spectrum'}})
            viewer.zoomTo()
            viewers.append(viewer)

        return viewers

    @staticmethod
    def create_overlay_view(pdb_contents: List[Tuple[str, str]],
                           width: int = 800,
                           height: int = 600) -> py3Dmol.view:
        """
        겹쳐서 비교 뷰

        Args:
            pdb_contents: [(이름, PDB 내용), ...] 리스트
            width: 뷰어 너비
            height: 뷰어 높이

        Returns:
            py3Dmol viewer 객체
        """
        viewer = py3Dmol.view(width=width, height=height)

        # 각 구조를 다른 색상으로 추가
        colors = ['cyan', 'magenta', 'yellow', 'green', 'orange', 'purple']

        for i, (name, pdb_content) in enumerate(pdb_contents):
            viewer.addModel(pdb_content, 'pdb')
            color = colors[i % len(colors)]
            viewer.setStyle({'model': i}, {'cartoon': {'color': color}})

        viewer.zoomTo()
        return viewer


class StylePresets:
    """
    사전 정의된 스타일 프리셋
    """

    @staticmethod
    def apply_publication_style(viewer: py3Dmol.view) -> py3Dmol.view:
        """
        출판용 스타일

        Args:
            viewer: py3Dmol viewer 객체

        Returns:
            스타일 적용된 viewer
        """
        viewer.setStyle({'cartoon': {'color': 'spectrum', 'thickness': 0.8}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        return viewer

    @staticmethod
    def apply_presentation_style(viewer: py3Dmol.view) -> py3Dmol.view:
        """
        프레젠테이션용 스타일

        Args:
            viewer: py3Dmol viewer 객체

        Returns:
            스타일 적용된 viewer
        """
        viewer.setStyle({'cartoon': {'color': 'spectrum', 'thickness': 1.0}})
        viewer.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
        viewer.setBackgroundColor('#f0f0f0')
        viewer.zoomTo()
        return viewer

    @staticmethod
    def apply_analysis_style(viewer: py3Dmol.view) -> py3Dmol.view:
        """
        분석용 스타일

        Args:
            viewer: py3Dmol viewer 객체

        Returns:
            스타일 적용된 viewer
        """
        viewer.setStyle({'stick': {'colorscheme': 'amino'}})
        viewer.setBackgroundColor('black')
        viewer.zoomTo()
        return viewer


class StructureAnnotator:
    """
    구조 주석 도구
    """

    @staticmethod
    def add_residue_labels(viewer: py3Dmol.view,
                          residue_numbers: List[int]) -> py3Dmol.view:
        """
        잔기 레이블 추가

        Args:
            viewer: py3Dmol viewer 객체
            residue_numbers: 레이블을 추가할 잔기 번호 리스트

        Returns:
            레이블 추가된 viewer
        """
        for resi in residue_numbers:
            viewer.addLabel(
                str(resi),
                {'resi': resi},
                {'fontSize': 12, 'fontColor': 'black', 'backgroundColor': 'white'}
            )

        return viewer

    @staticmethod
    def highlight_motif(viewer: py3Dmol.view,
                       start: int,
                       end: int,
                       color: str = 'red') -> py3Dmol.view:
        """
        모티프 영역 하이라이트

        Args:
            viewer: py3Dmol viewer 객체
            start: 시작 잔기 번호
            end: 끝 잔기 번호
            color: 하이라이트 색상

        Returns:
            하이라이트 추가된 viewer
        """
        viewer.addStyle(
            {'resi': list(range(start, end + 1))},
            {'cartoon': {'color': color, 'thickness': 1.2}}
        )

        return viewer


if __name__ == '__main__':
    # 테스트
    print("=== 3D Visualizer Test ===\n")

    # 샘플 PDB 내용 (매우 짧은 구조)
    sample_pdb = """ATOM      1  N   ALA A   1      11.104  12.766  13.756  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639  13.954  13.102  1.00  0.00           C
ATOM      3  C   ALA A   1      10.751  15.157  13.417  1.00  0.00           C
ATOM      4  O   ALA A   1      10.068  15.185  14.443  1.00  0.00           O
ATOM      5  N   ARG A   2      10.759  16.168  12.562  1.00  0.00           N
ATOM      6  CA  ARG A   2       9.965  17.383  12.775  1.00  0.00           C
ATOM      7  C   ARG A   2      10.619  18.558  12.046  1.00  0.00           C
ATOM      8  O   ARG A   2      11.840  18.636  11.913  1.00  0.00           O
END
"""

    print("1. Create viewer and load PDB")
    viewer = Py3DmolViewer(width=600, height=400)
    viewer.load_pdb_string(sample_pdb)
    print("   Loaded PDB structure")

    print("\n2. Apply cartoon style with spectrum coloring")
    viewer.set_style('cartoon', 'spectrum')
    viewer.render()
    print("   Applied cartoon style")

    print("\n3. Highlight specific residue")
    viewer.highlight_residue(1, color='red')
    print("   Highlighted residue 1")

    print("\n4. Get HTML representation")
    html = viewer.get_html()
    print(f"   Generated HTML ({len(html)} characters)")

    print("\n5. Create structure comparator")
    comparator = StructureComparator()
    print("   Comparator initialized")

    print("\n6. Apply publication style preset")
    preset_viewer = py3Dmol.view(width=600, height=400)
    preset_viewer.addModel(sample_pdb, 'pdb')
    StylePresets.apply_publication_style(preset_viewer)
    print("   Applied publication style")

    print("\n[OK] 3D Visualizer test complete!")
    print("\nNote: Viewers created but not displayed (use .show() in Jupyter)")
    print("      HTML can be saved to file or embedded in web pages")
