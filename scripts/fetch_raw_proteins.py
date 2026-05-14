"""
원료 단백질 자동 다운로드 (UniProt REST API 필터링)

회사 공정 자료가 확보된 제품들의 원료 단백질을 UniProt 검색 API에서
다중 필터(종 + Swiss-Prot 검수 + 종자 저장단백질 키워드)로 자동 추출합니다.

대상 제품:
  대두 (Glycine max):       SOY-1, SOY-N+, SOY-L, SOY-P, SOY-B, SOY-BIO, SOY-BIO N50
  쌀 (Oryza sativa jap.):   RICE-1, RICE-BIO
  밀 (Triticum aestivum):   WHEAT-1, WHEAT-BIO
  완두 (Pisum sativum):     PEA-1, PEA-BIO

제외:
  PPR Type4 (포크 펩톤) — 공정 정보 미확정으로 보류

사용법:
  python scripts/fetch_raw_proteins.py
"""

import json
import time
import ssl
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime


# ─── 원료별 UniProt 쿼리 정의 ──────────────────────────────
QUERIES = {
    # 모든 원료는 종자/곡립에서 추출하는 단백질이므로 동일 필터 적용:
    #   KW-0758 (Seed storage protein) - 주 storage protein
    #   KW-0020 (Allergen)             - Lectin, RAG 등 종자 알레르겐
    #   KW-0646 (Protease inhibitor)   - KTI, BBI, ATI 등 (펩톤에 다량 존재)

    "soy": {
        "korean_name": "대두",
        "scientific_name": "Glycine max",
        "products": ["SOY-1", "SOY-N+", "SOY-L", "SOY-P", "SOY-B",
                     "SOY-BIO", "SOY-BIO N50"],
        "query": "(organism_id:3847) AND (reviewed:true) AND "
                 "(keyword:KW-0758 OR keyword:KW-0020 OR keyword:KW-0646)",
        "note": "Glycinin, β-Conglycinin + Lipoxygenase, KTI, Lectin, Urease"
    },

    "rice": {
        "korean_name": "쌀 (japonica)",
        "scientific_name": "Oryza sativa subsp. japonica",
        "products": ["RICE-1", "RICE-BIO"],
        "query": "(organism_id:39947) AND (reviewed:true) AND "
                 "(keyword:KW-0758 OR keyword:KW-0020 OR keyword:KW-0646)",
        "note": "Glutelin, Prolamin + BBI, α-amylase inhibitor, Cysteine inhibitors"
    },

    "wheat": {
        "korean_name": "밀 (Vital Wheat Gluten)",
        "scientific_name": "Triticum aestivum",
        "products": ["WHEAT-1", "WHEAT-BIO"],
        # 밀은 키워드 매핑 누락 케이스가 있어 단백질명 OR 추가
        "query": "(organism_id:4565) AND (reviewed:true) AND "
                 "(keyword:KW-0758 OR keyword:KW-0020 OR keyword:KW-0646 OR "
                 "protein_name:gliadin OR protein_name:glutenin)",
        "note": "Gliadin, Glutenin + ATI (CM2/CM3/CM16), β-amylase, Serpin"
    },

    "pea": {
        "korean_name": "완두",
        "scientific_name": "Pisum sativum",
        "products": ["PEA-1", "PEA-BIO"],
        "query": "(organism_id:3888) AND (reviewed:true) AND "
                 "(keyword:KW-0758 OR keyword:KW-0020 OR keyword:KW-0646)",
        "note": "Legumin, Vicilin + Kunitz inhibitor, Lipid-transfer protein"
    },
}

# 결과 크기 제한 (한 원료당 최대 N개)
MAX_RESULTS_PER_MATERIAL = 100

# 너무 작은 단편 제외 (50 AA 미만 제외)
MIN_LENGTH = 50


# ─── HTTP 유틸 ──────────────────────────────────────────────
def http_get(url: str, timeout: int = 25) -> str:
    """SSL 검증 비활성 HTTPS GET (UniProt 호환)"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'peptide-structure-analyzer/2.0 (research)'}
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
        return r.read().decode('utf-8')


# ─── UniProt 검색 ───────────────────────────────────────────
def search_uniprot(query: str, size: int = 50):
    """검색 쿼리 → (FASTA 텍스트, JSON 메타데이터)"""
    encoded = urllib.parse.quote(query)

    fasta_url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query={encoded}&format=fasta&size={size}"
    )
    json_url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query={encoded}&format=json&size={size}"
        f"&fields=accession,id,protein_name,organism_name,length,"
        f"keyword,annotation_score,sequence"
    )

    fasta_text = http_get(fasta_url)
    json_text = http_get(json_url)
    json_data = json.loads(json_text)

    return fasta_text, json_data


def parse_fasta_bulk(fasta_text: str) -> dict:
    """Multi-FASTA → {accession: (header_line, sequence)}"""
    results = {}
    current_id = None
    current_header = None
    current_seq = []

    for line in fasta_text.split('\n'):
        line = line.rstrip()
        if line.startswith('>'):
            if current_id:
                results[current_id] = (current_header, ''.join(current_seq))
            current_header = line
            # UniProt FASTA: >sp|P04776|GLYG1_SOYBN Glycinin...
            parts = line[1:].split('|')
            current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line)

    if current_id:
        results[current_id] = (current_header, ''.join(current_seq))

    return results


def safe_filename(name: str, max_len: int = 60) -> str:
    """파일명 안전화"""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789_-.")
    safe = ''.join(c if c in allowed else '_' for c in name)
    # 연속 언더스코어 제거
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')[:max_len]


# ─── 메인 ───────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).parent.parent / "data" / "raw_proteins"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Raw Protein Auto-Download (UniProt REST API + Multi-Filter)")
    print("=" * 72)
    print(f" Filters: Swiss-Prot reviewed + Seed storage keyword + min {MIN_LENGTH} AA")
    print()

    summary = {
        "fetched_at": datetime.now().isoformat(),
        "source": "UniProt REST API (https://rest.uniprot.org)",
        "filters_applied": {
            "reviewed_only": True,
            "min_length": MIN_LENGTH,
            "max_per_material": MAX_RESULTS_PER_MATERIAL,
        },
        "materials": {}
    }

    grand_total = 0
    grand_residues = 0

    for material, info in QUERIES.items():
        material_dir = base_dir / material
        material_dir.mkdir(exist_ok=True)

        print(f"[{material.upper():6s}] {info['korean_name']} "
              f"({info['scientific_name']})")
        print(f"  Products: {', '.join(info['products'])}")
        print(f"  Query   : {info['query']}")
        print(f"  Searching UniProt... ", end='', flush=True)

        try:
            fasta_text, json_data = search_uniprot(
                info['query'], size=MAX_RESULTS_PER_MATERIAL
            )
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.reason}")
            continue
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")
            continue

        fasta_map = parse_fasta_bulk(fasta_text)
        results = json_data.get('results', [])

        print(f"Found {len(results)} candidates")

        proteins_data = []
        for entry in results:
            acc = entry.get('primaryAccession', '')
            if not acc or acc not in fasta_map:
                continue

            header, seq = fasta_map[acc]

            # 최소 길이 필터
            if len(seq) < MIN_LENGTH:
                continue

            # 단백질 이름 추출
            try:
                full_name = (entry['proteinDescription']['recommendedName']
                             ['fullName']['value'])
            except (KeyError, TypeError):
                full_name = entry.get('uniProtkbId', acc)

            # 개별 FASTA 저장
            fname = f"{acc}_{safe_filename(full_name)}.fasta"
            fasta_path = material_dir / fname
            fasta_path.write_text(f"{header}\n{seq}\n", encoding='utf-8')

            # 메타데이터 수집
            score = entry.get('annotationScore', 0)
            keywords = [k.get('name', '') for k in entry.get('keywords', [])]

            proteins_data.append({
                "uniprot_id": acc,
                "uniprot_name": entry.get('uniProtkbId', ''),
                "full_name": full_name,
                "length": len(seq),
                "annotation_score": score,
                "keywords": keywords[:6],
                "fasta_file": str(fasta_path.relative_to(base_dir.parent)),
                "sequence": seq,
            })

        # 길이 내림차순 정렬 (큰 단백질 = 보통 핵심 storage protein)
        proteins_data.sort(key=lambda x: x['length'], reverse=True)

        # 콘솔 출력
        for p in proteins_data:
            kw_str = ', '.join(p['keywords'][:3])[:50]
            name_str = p['full_name'][:45]
            print(f"    {p['uniprot_id']:8s} {p['length']:5d} AA  "
                  f"score={p['annotation_score']:.1f}  {name_str}")

        summary["materials"][material] = {
            "korean_name": info["korean_name"],
            "scientific_name": info["scientific_name"],
            "products": info["products"],
            "uniprot_query": info["query"],
            "note": info["note"],
            "n_proteins": len(proteins_data),
            "total_residues": sum(p["length"] for p in proteins_data),
            "proteins": proteins_data
        }

        grand_total += len(proteins_data)
        grand_residues += sum(p["length"] for p in proteins_data)
        print()
        time.sleep(0.5)  # UniProt 부담 방지

    # 인덱스 저장
    index_path = base_dir / "raw_proteins_index.json"
    index_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print("=" * 72)
    print(f" Total: {grand_total} proteins, {grand_residues:,} residues")
    print(f" Index: {index_path.relative_to(Path(__file__).parent.parent)}")
    print(f" FASTA: data/raw_proteins/{{material}}/*.fasta")
    print("=" * 72)


if __name__ == "__main__":
    main()
