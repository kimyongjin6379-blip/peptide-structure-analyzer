#!/usr/bin/env python3
"""
Comprehensive Bioactive Peptide Database Builder
=================================================
Scrapes BIOPEP-UWM and combines with existing data to create
a large-scale bioactive peptide database for peptone research.

Sources:
1. BIOPEP-UWM (https://biochemia.uwm.edu.pl/biopep/) - 5,601 food-derived bioactive peptides
2. Existing local DB (bioactive_peptide_db.json) - 52 curated motifs with IC50/source

Usage:
    python scripts/build_bioactive_db.py

Output:
    data/bioactive_peptide_db_comprehensive.json
"""

import json
import os
import re
import sys
import time
import logging
from pathlib import Path
import ssl
from urllib.request import urlopen, Request
from html.parser import HTMLParser
from collections import defaultdict

# Disable SSL verification for academic servers with certificate issues
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "bioactive_peptide_db_comprehensive.json"
CACHE_DIR = DATA_DIR / "raw_db" / "biopep_cache"

BIOPEP_LIST_URL = "https://biochemia.uwm.edu.pl/biopep/peptide_data.php?pageNum_result1={page}"
BIOPEP_DETAIL_URL = "https://biochemia.uwm.edu.pl/biopep/peptide_data_page1.php?zm_ID={id}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# BIOPEP activity code mapping
ACTIVITY_MAP = {
    "ace": "ACE_inhibitor",
    "ACE inhibitor": "ACE_inhibitor",
    "antioxidative": "antioxidant",
    "antioxidant": "antioxidant",
    "hypotensive": "antihypertensive",
    "antihypertensive": "antihypertensive",
    "antibacterial": "antibacterial",
    "antimicrobial": "antimicrobial",
    "anticancer": "anticancer",
    "antiamnestic": "antiamnestic",
    "anti-inflammatory": "anti_inflammatory",
    "antithrombotic": "antithrombotic",
    "antiviral": "antiviral",
    "DPP IV inhibitor": "DPP_IV_inhibitor",
    "DPP-IV inhibitor": "DPP_IV_inhibitor",
    "dipeptidyl peptidase IV inhibitor": "DPP_IV_inhibitor",
    "immunomodulating": "immunomodulatory",
    "opioid": "opioid",
    "opioid agonist": "opioid",
    "opioid antagonist": "opioid",
    "neuropeptide": "neuropeptide",
    "regulating": "regulatory",
    "stimulating": "stimulating",
    "inhibitor": "enzyme_inhibitor",
    "renin inhibitor": "renin_inhibitor",
    "alpha-glucosidase inhibitor": "alpha_glucosidase_inhibitor",
    "antidiabetic": "antidiabetic",
    "celiac toxic": "celiac_toxic",
    "bitter": "bitter",
    "umami": "umami",
    "sweet": "sweet",
    "salty": "salty",
    "sour": "sour",
}


class BiopepListParser(HTMLParser):
    """Parse BIOPEP-UWM peptide list pages"""

    def __init__(self):
        super().__init__()
        self.peptides = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row = []
        self.current_cell = ""
        self.table_count = 0
        self.row_count = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "table":
            self.table_count += 1
        if tag == "tr":
            self.in_row = True
            self.current_row = []
            self.row_count += 1
        if tag == "td":
            self.in_cell = True
            self.current_cell = ""
        if tag == "a" and self.in_cell:
            href = attrs_dict.get("href", "")
            if "zm_ID=" in href:
                match = re.search(r'zm_ID=(\d+)', href)
                if match:
                    self.current_cell += f"[ID:{match.group(1)}]"

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
        if tag == "tr" and self.in_row:
            self.in_row = False
            if len(self.current_row) >= 5:
                self._process_row(self.current_row)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data

    def _process_row(self, row):
        """Extract peptide data from table row"""
        # Actual columns: [Link+ID] [ID_num] [Name] [Sequence] [Chem Mass] [Monois Mass] [EC50/IC50]
        try:
            # Extract ID from link in first cell
            id_match = re.search(r'\[ID:(\d+)\]', row[0])
            peptide_id = int(id_match.group(1)) if id_match else None

            if peptide_id is None:
                return

            entry = {
                "id": peptide_id,
                "name": row[2].strip() if len(row) > 2 else "",
                "sequence": row[3].strip().upper() if len(row) > 3 else "",
                "chemical_mass": self._parse_float(row[4]) if len(row) > 4 else None,
                "monoisotopic_mass": self._parse_float(row[5]) if len(row) > 5 else None,
            }

            # Parse activity/EC50 from 7th column
            if len(row) > 6:
                activity_text = row[6].strip()
                ec50_match = re.search(r'([\d.]+)\s*(EC50|IC50|EC|IC)', activity_text)
                if ec50_match:
                    val = float(ec50_match.group(1))
                    entry["ec50_ic50"] = val if val > 0 else None
                    entry["ec50_ic50_type"] = "EC50" if "EC" in ec50_match.group(2) else "IC50"

            # Validate sequence (only standard amino acids)
            if entry["sequence"] and re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', entry["sequence"]):
                self.peptides.append(entry)

        except (ValueError, IndexError) as e:
            pass

    @staticmethod
    def _parse_float(text):
        try:
            val = float(re.sub(r'[^\d.]', '', text))
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None


class BiopepDetailParser(HTMLParser):
    """Parse BIOPEP-UWM individual peptide detail pages"""

    def __init__(self):
        super().__init__()
        self.activities = []
        self.references = []
        self.in_link = False
        self.current_text = ""
        self.all_text = []
        self.in_td = False
        self.td_texts = []
        self.current_td = ""

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.in_td = True
            self.current_td = ""
        if tag == "a":
            self.in_link = True

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_td = False
            self.td_texts.append(self.current_td.strip())
        if tag == "a":
            self.in_link = False

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.all_text.append(text)
        if self.in_td:
            self.current_td += data

    def get_activities(self):
        """Extract activity types from page text"""
        activities = []
        for text in self.td_texts:
            text_lower = text.lower().strip()
            for key, normalized in ACTIVITY_MAP.items():
                if key.lower() in text_lower:
                    if normalized not in activities:
                        activities.append(normalized)
        return activities


def fetch_url(url, retries=3, delay=1.0):
    """Fetch URL with retry logic"""
    for attempt in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
                return resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                logger.warning(f"Failed to fetch {url}: {e}")
                return None


def scrape_biopep_list_page(page_num):
    """Scrape a single BIOPEP-UWM list page"""
    cache_file = CACHE_DIR / f"list_page_{page_num}.html"

    if cache_file.exists():
        html = cache_file.read_text(encoding='utf-8')
    else:
        url = BIOPEP_LIST_URL.format(page=page_num)
        html = fetch_url(url)
        if html:
            cache_file.write_text(html, encoding='utf-8')
        else:
            return []

    parser = BiopepListParser()
    parser.feed(html)
    return parser.peptides


def scrape_biopep_detail(peptide_id):
    """Scrape individual peptide detail page for activity info"""
    cache_file = CACHE_DIR / f"detail_{peptide_id}.html"

    if cache_file.exists():
        html = cache_file.read_text(encoding='utf-8')
    else:
        url = BIOPEP_DETAIL_URL.format(id=peptide_id)
        html = fetch_url(url)
        if html:
            cache_file.write_text(html, encoding='utf-8')
        else:
            return []
        time.sleep(0.3)  # Be polite to server

    parser = BiopepDetailParser()
    parser.feed(html)
    return parser.get_activities()


def scrape_biopep_all():
    """Scrape all BIOPEP-UWM peptides"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_peptides = {}
    total_pages = 374

    logger.info(f"Scraping BIOPEP-UWM: {total_pages} pages...")

    for page in range(total_pages):
        peptides = scrape_biopep_list_page(page)
        for p in peptides:
            pid = p["id"]
            if pid not in all_peptides:
                all_peptides[pid] = p

        if (page + 1) % 20 == 0:
            logger.info(f"  Progress: {page + 1}/{total_pages} pages, {len(all_peptides)} peptides")
        time.sleep(0.2)  # Rate limiting

    logger.info(f"List scraping complete: {len(all_peptides)} peptides")

    # Scrape detail pages for activity classification (sample first 200 for speed)
    # Full detail scraping can be done later with --full flag
    detail_limit = len(all_peptides) if "--full" in sys.argv else min(200, len(all_peptides))
    logger.info(f"Scraping detail pages for activity classification ({detail_limit} peptides)...")
    logger.info("  (Use --full flag for all peptides, takes ~30 min)")
    detail_count = 0
    for pid, pdata in list(all_peptides.items())[:detail_limit]:
        activities = scrape_biopep_detail(pid)
        if activities:
            pdata["activities"] = activities
        detail_count += 1
        if detail_count % 50 == 0:
            logger.info(f"  Detail progress: {detail_count}/{detail_limit}")

    # Infer activities from peptide names for those without detail data
    name_activity_keywords = {
        "ACE": "ACE_inhibitor", "ace inhibit": "ACE_inhibitor",
        "antioxid": "antioxidant", "anti-oxid": "antioxidant",
        "antihypertens": "antihypertensive", "hypotens": "antihypertensive",
        "antibacter": "antibacterial", "antimicrob": "antimicrobial",
        "anticancer": "anticancer", "antitumor": "anticancer",
        "anti-inflamm": "anti_inflammatory",
        "antithrombot": "antithrombotic",
        "antiviral": "antiviral",
        "DPP": "DPP_IV_inhibitor", "dipeptidyl": "DPP_IV_inhibitor",
        "opioid": "opioid", "hemorphin": "opioid",
        "immunomod": "immunomodulatory",
        "neuropeptide": "neuropeptide",
        "antidiabet": "antidiabetic", "glucosidase": "alpha_glucosidase_inhibitor",
        "renin": "renin_inhibitor",
        "bitter": "bitter", "umami": "umami", "sweet": "sweet",
        "celiac": "celiac_toxic", "coeliac": "celiac_toxic",
    }
    inferred = 0
    for pdata in all_peptides.values():
        if not pdata.get("activities"):
            name_lower = pdata.get("name", "").lower()
            acts = []
            for keyword, activity in name_activity_keywords.items():
                if keyword.lower() in name_lower and activity not in acts:
                    acts.append(activity)
            if acts:
                pdata["activities"] = acts
                inferred += 1
    logger.info(f"  Inferred activities from names: {inferred} peptides")

    return list(all_peptides.values())


def load_existing_db():
    """Load the existing curated bioactive peptide database"""
    db_file = DATA_DIR / "bioactive_peptide_db.json"
    if db_file.exists():
        with open(db_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("motifs", [])
    return []


def normalize_activity(activity_str):
    """Normalize activity type string"""
    activity_lower = activity_str.lower().strip()
    for key, normalized in ACTIVITY_MAP.items():
        if key.lower() in activity_lower:
            return normalized
    return activity_str.lower().replace(" ", "_").replace("-", "_")


def merge_databases(biopep_peptides, existing_motifs):
    """Merge BIOPEP-UWM data with existing curated database"""
    merged = {}

    # Add existing curated data first (higher quality)
    for motif in existing_motifs:
        seq = motif["sequence"].upper().strip()
        entry = {
            "sequence": seq,
            "length": len(seq),
            "activities": [normalize_activity(motif.get("activity", ""))],
            "description": motif.get("description", ""),
            "source_protein": motif.get("source", ""),
            "references": motif.get("references", []),
            "ic50": motif.get("IC50", None),
            "db_sources": ["curated"],
            "molecular_weight": None,
        }
        merged[seq] = entry

    # Add BIOPEP-UWM data
    for bp in biopep_peptides:
        seq = bp["sequence"].upper().strip()
        if not seq:
            continue

        if seq in merged:
            # Merge activities
            existing = merged[seq]
            new_activities = bp.get("activities", [])
            for act in new_activities:
                if act not in existing["activities"]:
                    existing["activities"].append(act)
            if "BIOPEP-UWM" not in existing["db_sources"]:
                existing["db_sources"].append("BIOPEP-UWM")
            if bp.get("chemical_mass") and not existing["molecular_weight"]:
                existing["molecular_weight"] = bp["chemical_mass"]
            if bp.get("ec50_ic50") and not existing["ic50"]:
                existing["ic50"] = f"{bp['ec50_ic50']} uM ({bp.get('ec50_ic50_type', 'IC50')})"
            existing["biopep_id"] = bp.get("id")
        else:
            entry = {
                "sequence": seq,
                "length": len(seq),
                "activities": bp.get("activities", ["unknown"]),
                "description": bp.get("name", ""),
                "source_protein": "",
                "references": [],
                "ic50": None,
                "db_sources": ["BIOPEP-UWM"],
                "molecular_weight": bp.get("chemical_mass"),
                "biopep_id": bp.get("id"),
            }
            if bp.get("ec50_ic50"):
                entry["ic50"] = f"{bp['ec50_ic50']} uM ({bp.get('ec50_ic50_type', 'IC50')})"
            merged[seq] = entry

    return merged


def build_comprehensive_db():
    """Main function to build the comprehensive database"""
    logger.info("=" * 60)
    logger.info("Comprehensive Bioactive Peptide Database Builder")
    logger.info("=" * 60)

    # Step 1: Load existing curated DB
    logger.info("\n[1/4] Loading existing curated database...")
    existing = load_existing_db()
    logger.info(f"  Loaded {len(existing)} curated motifs")

    # Step 2: Scrape BIOPEP-UWM
    logger.info("\n[2/4] Scraping BIOPEP-UWM database...")
    biopep_peptides = scrape_biopep_all()
    logger.info(f"  Scraped {len(biopep_peptides)} peptides from BIOPEP-UWM")

    # Step 3: Merge databases
    logger.info("\n[3/4] Merging databases...")
    merged = merge_databases(biopep_peptides, existing)
    logger.info(f"  Merged database: {len(merged)} unique peptides")

    # Step 4: Build output
    logger.info("\n[4/4] Building output database...")

    # Activity statistics
    activity_counts = defaultdict(int)
    for entry in merged.values():
        for act in entry["activities"]:
            activity_counts[act] += 1

    # Build final structure
    output = {
        "metadata": {
            "name": "Comprehensive Bioactive Peptide Database",
            "version": "2.0",
            "description": "Integrated database of food-derived bioactive peptides",
            "sources": [
                "BIOPEP-UWM (University of Warmia and Mazury)",
                "Curated literature database"
            ],
            "total_peptides": len(merged),
            "activity_types": len(activity_counts),
            "activity_statistics": dict(sorted(activity_counts.items(), key=lambda x: -x[1])),
            "build_date": time.strftime("%Y-%m-%d"),
        },
        "peptides": sorted(merged.values(), key=lambda x: x["sequence"]),
    }

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Database saved to: {OUTPUT_FILE}")
    logger.info(f"Total peptides: {len(merged)}")
    logger.info(f"Activity types: {len(activity_counts)}")
    logger.info(f"\nTop activities:")
    for act, count in sorted(activity_counts.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"  {act}: {count}")
    logger.info(f"{'=' * 60}")

    return output


if __name__ == "__main__":
    build_comprehensive_db()
