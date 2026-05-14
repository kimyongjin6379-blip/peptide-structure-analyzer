"""
Enzyme Processor — 효소 공정 메타데이터 관리

회사 펩톤 제품별 효소 공정 정보를 로드하고 조회하는 모듈.
향후 in silico digestion 모듈의 핵심 입력으로 사용됩니다.

사용 예:
    from enzyme_processor import EnzymeProcessor

    ep = EnzymeProcessor()
    process = ep.get_process("SOY-1")
    print(process.enzymes_used)
    print(process.expected_peptide_size_range())
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ─── 데이터 클래스 ──────────────────────────────────────────
@dataclass
class EnzymeInfo:
    """효소 정보"""
    name: str
    manufacturer: str
    source: str
    type: str
    specificity: str
    cleaves_after: Any  # list 또는 str
    exo_activity: bool = False
    exo_targets: List[str] = field(default_factory=list)
    exo_terminal: str = "none"
    optimum_temp_c: List[int] = field(default_factory=list)
    function_summary: str = ""
    raw: Dict = field(default_factory=dict)


@dataclass
class ProcessStep:
    """공정 단계 (전처리, 메인 분해, 후처리)"""
    enzymes: List[Dict]  # [{"enzyme": "Alcalase", "concentration_pct": 0.1}]
    temperature_c: Optional[float] = None
    duration_min: Optional[float] = None
    duration_hours: Optional[float] = None
    extra: Dict = field(default_factory=dict)


@dataclass
class PeptoneProcess:
    """단일 펩톤 제품의 전체 공정"""
    product_id: str
    category: str  # "1st_gen", "2nd_gen_bio"
    raw_material_id: str  # "soy", "rice", "wheat", "pea"
    raw_material_form: str
    raw_concentration_pct: float
    pretreatment: Optional[ProcessStep] = None
    main_hydrolysis: Optional[ProcessStep] = None
    post_treatment: Optional[Dict] = None
    additives: Optional[List[Dict]] = None
    based_on: Optional[str] = None  # BIO 제품의 베이스 제품
    is_blend: bool = False
    notes: str = ""

    @property
    def enzymes_used(self) -> List[str]:
        """이 공정에 사용된 모든 효소 이름"""
        enzymes = set()
        for step in [self.pretreatment, self.main_hydrolysis]:
            if step:
                for e in step.enzymes:
                    enzymes.add(e["enzyme"])
        return sorted(enzymes)

    @property
    def has_uf(self) -> bool:
        """UF 후처리 여부"""
        if not self.post_treatment:
            return False
        return self.post_treatment.get("type", "").startswith("UF")

    @property
    def uf_cutoff_kda(self) -> Optional[float]:
        """UF 컷오프 (kDa)"""
        if not self.has_uf:
            return None
        return self.post_treatment.get("uf_cutoff_kda")

    def expected_peptide_size_range(self) -> Dict[str, float]:
        """공정 특성 기반 예상 펩타이드 크기 범위 추정"""
        # Alcalase 단독: 큰 펩타이드 (10~30 AA)
        # Alcalase + ZF101/Flavourzyme: 짧은 펩타이드 (3~10 AA) + 유리 AA
        # + UF 3K: ≤ ~27 AA로 제한
        enzymes = self.enzymes_used
        has_exo = any(e in enzymes for e in ["ZF101", "Flavourzyme"])

        if self.is_blend:
            return {"min_aa": 1, "max_aa": 999, "note": "Blending, 다양한 크기"}

        if not enzymes and self.based_on:
            # BIO 제품 (UF만 적용)
            base_range = {"min_aa": 3, "max_aa": 10}
            if self.has_uf:
                base_range["max_aa"] = min(base_range["max_aa"], 27)
                base_range["note"] = f"BIO (UF {self.uf_cutoff_kda}K Da)"
            return base_range

        if has_exo:
            # ZF101/Flavourzyme 포함
            r = {"min_aa": 3, "max_aa": 10, "note": "endo + exo (debittering)"}
        else:
            # Alcalase 단독
            r = {"min_aa": 10, "max_aa": 30, "note": "Alcalase 단독, 큰 펩타이드"}

        if self.has_uf:
            r["max_aa"] = min(r["max_aa"], 27)
            r["note"] += f" + UF {self.uf_cutoff_kda}K"

        return r


# ─── 메인 클래스 ────────────────────────────────────────────
class EnzymeProcessor:
    """효소 공정 메타데이터 관리자"""

    def __init__(self, json_path: Optional[Path] = None):
        if json_path is None:
            json_path = Path(__file__).parent.parent / "data" / "enzyme_processes.json"

        with open(json_path, encoding='utf-8') as f:
            self.data = json.load(f)

        self.enzymes: Dict[str, EnzymeInfo] = self._load_enzymes()
        self.processes: Dict[str, PeptoneProcess] = self._load_processes()

    def _load_enzymes(self) -> Dict[str, EnzymeInfo]:
        """효소 정보 로드"""
        enzymes = {}
        for name, info in self.data.get("enzymes", {}).items():
            enzymes[name] = EnzymeInfo(
                name=name,
                manufacturer=info.get("manufacturer", ""),
                source=info.get("source", ""),
                type=info.get("type", ""),
                specificity=info.get("specificity", ""),
                cleaves_after=info.get("cleaves_after", []),
                exo_activity=info.get("exo_activity", False),
                exo_targets=info.get("exo_targets", []),
                exo_terminal=info.get("exo_terminal", "none"),
                optimum_temp_c=info.get("optimum_temp_c", []),
                function_summary=info.get("function_summary", ""),
                raw=info
            )
        return enzymes

    def _load_processes(self) -> Dict[str, PeptoneProcess]:
        """제품 공정 로드"""
        processes = {}
        for pid, p in self.data.get("products", {}).items():
            raw = p.get("raw_material", {})

            # ProcessStep 파싱
            def parse_step(step_data):
                if not step_data:
                    return None
                # 전처리는 단일 효소, main_hydrolysis는 enzymes 리스트
                if "enzyme" in step_data:
                    enzymes = [{
                        "enzyme": step_data["enzyme"],
                        "concentration_pct": step_data.get("concentration_pct")
                    }]
                else:
                    enzymes = step_data.get("enzymes", [])
                return ProcessStep(
                    enzymes=enzymes,
                    temperature_c=step_data.get("temperature_c"),
                    duration_min=step_data.get("duration_min"),
                    duration_hours=step_data.get("duration_hours"),
                    extra={k: v for k, v in step_data.items()
                           if k not in ["enzyme", "enzymes", "concentration_pct",
                                        "temperature_c", "duration_min",
                                        "duration_hours"]}
                )

            processes[pid] = PeptoneProcess(
                product_id=pid,
                category=p.get("category", "unknown"),
                raw_material_id=raw.get("id", ""),
                raw_material_form=raw.get("form", ""),
                raw_concentration_pct=raw.get("concentration_pct", 0),
                pretreatment=parse_step(p.get("pretreatment")),
                main_hydrolysis=parse_step(p.get("main_hydrolysis")),
                post_treatment=p.get("post_treatment"),
                additives=p.get("additives"),
                based_on=p.get("based_on"),
                is_blend=p.get("is_blend", False),
                notes=p.get("notes", "")
            )
        return processes

    # ─── 조회 메서드 ──────────────────────────────────────
    def get_process(self, product_id: str) -> Optional[PeptoneProcess]:
        """특정 제품의 공정 정보 반환"""
        return self.processes.get(product_id)

    def get_enzyme(self, enzyme_name: str) -> Optional[EnzymeInfo]:
        """특정 효소 정보 반환"""
        return self.enzymes.get(enzyme_name)

    def list_products(self) -> List[str]:
        """모든 제품 ID 리스트"""
        return list(self.processes.keys())

    def list_products_by_material(self, material: str) -> List[str]:
        """원료별 제품 리스트 (soy, rice, wheat, pea)"""
        return [pid for pid, p in self.processes.items()
                if p.raw_material_id == material]

    def list_products_by_category(self, category: str) -> List[str]:
        """카테고리별 제품 리스트 (1st_gen, 2nd_gen_bio)"""
        return [pid for pid, p in self.processes.items()
                if p.category == category]

    def get_bio_chain(self, bio_product: str) -> List[str]:
        """BIO 제품 → 베이스 제품 체인 반환 (예: SOY-BIO → SOY-1)"""
        chain = [bio_product]
        current = self.processes.get(bio_product)
        while current and current.based_on:
            chain.append(current.based_on)
            current = self.processes.get(current.based_on)
        return chain

    def summary_table(self) -> List[Dict]:
        """전체 제품 요약 테이블"""
        rows = []
        for pid, p in self.processes.items():
            size = p.expected_peptide_size_range()
            rows.append({
                "Product": pid,
                "Category": p.category,
                "Raw Material": p.raw_material_id,
                "Concentration": f"{p.raw_concentration_pct}%",
                "Enzymes": ", ".join(p.enzymes_used) or "—",
                "UF": f"{p.uf_cutoff_kda}K" if p.has_uf else "—",
                "Expected Size": f"{size['min_aa']}-{size['max_aa']} AA",
                "Note": p.notes[:50]
            })
        return rows


# ─── 셀프 테스트 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    ep = EnzymeProcessor()

    print("=" * 80)
    print(" Enzyme Process Database")
    print("=" * 80)
    print(f"\nLoaded: {len(ep.enzymes)} enzymes, {len(ep.processes)} products\n")

    # 효소 정보
    print("[ENZYMES]")
    for name, e in ep.enzymes.items():
        exo = f" + exo({','.join(e.exo_targets)})" if e.exo_activity else ""
        print(f"  {name:15s} {e.type:25s} {e.specificity}{exo}")

    # 전체 제품 요약
    print("\n[PRODUCTS]")
    for row in ep.summary_table():
        print(f"  {row['Product']:13s} {row['Raw Material']:6s} "
              f"{row['Enzymes']:30s} {row['UF']:5s} {row['Expected Size']}")

    # 원료별 그룹
    print("\n[BY MATERIAL]")
    for mat in ["soy", "rice", "wheat", "pea"]:
        products = ep.list_products_by_material(mat)
        print(f"  {mat:6s}: {', '.join(products)}")

    # BIO 체인
    print("\n[BIO CHAINS]")
    for bio in ep.list_products_by_category("2nd_gen_bio"):
        chain = ep.get_bio_chain(bio)
        print(f"  {' ← '.join(chain)}")

    # 특정 제품 상세
    print("\n[SAMPLE PROCESS: SOY-1]")
    p = ep.get_process("SOY-1")
    if p:
        print(f"  Raw      : {p.raw_material_form} ({p.raw_concentration_pct}%)")
        if p.pretreatment:
            e = p.pretreatment.enzymes[0]
            print(f"  Pretreat : {e['enzyme']} {e['concentration_pct']}% "
                  f"@ {p.pretreatment.temperature_c}°C × {p.pretreatment.duration_min}min")
        if p.main_hydrolysis:
            enz_str = " + ".join(
                f"{e['enzyme']} {e['concentration_pct']}%"
                for e in p.main_hydrolysis.enzymes
            )
            print(f"  Main     : {enz_str} @ {p.main_hydrolysis.temperature_c}°C "
                  f"× {p.main_hydrolysis.duration_hours}h")
        print(f"  Enzymes  : {', '.join(p.enzymes_used)}")
        size = p.expected_peptide_size_range()
        print(f"  Expected : {size['min_aa']}-{size['max_aa']} AA ({size.get('note', '')})")
