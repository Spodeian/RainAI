"""
Automated audio scraper and dataset ingest pipeline for diverse rain, storm, nature sounds,
OpenAIR / ARTE spatial Room Impulse Responses (RIRs), BigSoundBank CC0 audio, and Figshare rainfall datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import re
import json
import requests
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class LicenseVerifier:
    """
    Automated license compliance verifier ensuring all ingested audio is strictly
    compatible with open and commercial use (CC0, Public Domain, CC-BY, CC-BY-SA),
    explicitly rejecting NonCommercial (-NC) and NoDerivatives (-ND) clauses.
    """
    
    ALLOWED_KEYWORDS = {
        "cc0", "cc-0", "public domain", "publicdomain", "pd",
        "cc-by", "cc by", "cc-by-4.0", "cc-by-3.0", "cc-by-2.5", "cc-by-2.0",
        "cc-by-sa", "cc by-sa", "cc-by-sa-4.0", "cc-by-sa-3.0",
        "open access", "unlicense", "mit", "apache", "bsd"
    }

    DISALLOWED_PATTERNS = [
        r"\bnc\b",
        r"\bnon[- ]?commercial\b",
        r"\bnd\b",
        r"\bno[- ]?derivatives\b",
        r"\bsampling\+?\b",
        r"\bpersonal\b",
        r"\beditorial\b",
        r"\ball rights reserved\b"
    ]

    @classmethod
    def verify(cls, license_name: str) -> Tuple[bool, str]:
        """
        Validates if a license string allows commercial open use.
        Returns: (is_approved: bool, reason: str)
        """
        if not license_name:
            return False, "Empty or missing license description"

        clean = license_name.strip().lower()

        # 1. Check for hard disallow patterns
        for pat in cls.DISALLOWED_PATTERNS:
            if re.search(pat, clean):
                return False, f"Rejected: Contains restrictive clause matching pattern '{pat}' ({license_name})"

        # 2. Check for recognized allowed keywords
        for allowed in cls.ALLOWED_KEYWORDS:
            if allowed in clean:
                return True, f"Approved: Matches permissive open license '{allowed}'"

        # 3. Handle specific Creative Commons variations
        if "creative commons" in clean:
            if "attribution" in clean and "noncommercial" not in clean and "no-derivatives" not in clean:
                return True, f"Approved: Creative Commons Attribution ({license_name})"
            if "zero" in clean or "0" in clean:
                return True, "Approved: Creative Commons Zero (CC0)"

        return False, f"Rejected: Unknown or unverified license scheme ({license_name})"


@dataclass
class AudioDownloadItem:
    url: str
    target_filename: str
    category: str
    license: str
    description: str


# Curated public domain / Creative Commons CC-BY / CC0 audio sources for instant ingestion
CURATED_SOURCES = [
    # 1. Steady Rain & Downpours
    AudioDownloadItem(
        url="https://archive.org/download/RainSounds10Hours/RainSounds10Hours_vbr.mp3",
        target_filename="archive_rain_ambient.mp3",
        category="steady_rain",
        license="Public Domain",
        description="Continuous ambient rain recording"
    ),
    AudioDownloadItem(
        url="https://archive.org/download/ThunderstormSounds/ThunderstormSounds_vbr.mp3",
        target_filename="archive_thunderstorm.mp3",
        category="thunderstorm",
        license="Public Domain",
        description="Distant and rolling thunder with heavy rain"
    ),
    # 2. BigSoundBank CC0 Rain & Weather Recordings
    AudioDownloadItem(
        url="https://bigsoundbank.com/UPLOAD/mp3/0740.mp3",
        target_filename="bigsoundbank_rain_thunder_0740.mp3",
        category="heavy_rain_thunder",
        license="CC0",
        description="Heavy rain and thunderstorm recorded in stereo (BigSoundBank CC0)"
    ),
    AudioDownloadItem(
        url="https://bigsoundbank.com/UPLOAD/mp3/2719.mp3",
        target_filename="bigsoundbank_rain_downpour_2719.mp3",
        category="rain_downpour",
        license="CC0",
        description="Sudden torrential downpour on outdoor terrain (BigSoundBank CC0)"
    ),
    AudioDownloadItem(
        url="https://bigsoundbank.com/UPLOAD/mp3/1019.mp3",
        target_filename="bigsoundbank_rain_pavement_1019.mp3",
        category="surface_pavement",
        license="CC0",
        description="Steady rainfall impacting street pavement (BigSoundBank CC0)"
    ),
    AudioDownloadItem(
        url="https://bigsoundbank.com/UPLOAD/mp3/0124.mp3",
        target_filename="bigsoundbank_rain_window_0124.mp3",
        category="surface_window",
        license="CC0",
        description="Rain drops tapping on glass window pane (BigSoundBank CC0)"
    ),
    # 3. OpenAIR / Spatial Room Impulse Responses (for enclosure & distance modeling)
    AudioDownloadItem(
        url="https://www.openair.hosted.york.ac.uk/wp-content/uploads/2019/07/st_margarets_church_b_format.wav",
        target_filename="rir_stone_church_foa.wav",
        category="spatial_rir_stone",
        license="CC-BY-4.0",
        description="Stone room impulse response in 4-channel B-format FOA"
    ),
    AudioDownloadItem(
        url="https://www.openair.hosted.york.ac.uk/wp-content/uploads/2019/07/arthur_sykes_rymer_auditorium_b_format.wav",
        target_filename="rir_wood_auditorium_foa.wav",
        category="spatial_rir_wood",
        license="CC-BY-4.0",
        description="Wood auditorium impulse response in 4-channel B-format FOA"
    ),
    # 4. FSD50K Curated Surfaces & Side Sounds
    AudioDownloadItem(
        url="https://freesound.org/data/previews/343/343360_5121236-lq.mp3",
        target_filename="surface_rain_tin_roof.mp3",
        category="surface_tin",
        license="CC0",
        description="Rain drops striking corrugated metal tin roof"
    ),
    AudioDownloadItem(
        url="https://freesound.org/data/previews/518/518884_2402876-lq.mp3",
        target_filename="surface_rain_canvas_tent.mp3",
        category="surface_canvas",
        license="CC0",
        description="Rain drumming softly on canvas camping tent"
    ),
    AudioDownloadItem(
        url="https://freesound.org/data/previews/442/442907_6142149-lq.mp3",
        target_filename="side_sound_night_insects.mp3",
        category="side_insects",
        license="CC0",
        description="Night cicadas and crickets in humid rain forest"
    ),
    AudioDownloadItem(
        url="https://freesound.org/data/previews/360/360325_649468-lq.mp3",
        target_filename="side_sound_wood_fireplace.mp3",
        category="side_fireplace",
        license="CC0",
        description="Wood fireplace ember pops and gentle crackling"
    ),
]


class RainAudioIngester:
    """Ingests, downloads, and categorizes rain, atmospheric audio, and spatial impulse response datasets."""
    def __init__(self, target_dir: Path, freesound_api_key: Optional[str] = None):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.rirs_dir = self.target_dir / "rirs"
        self.rirs_dir.mkdir(parents=True, exist_ok=True)
        self.freesound_api_key = freesound_api_key or os.environ.get("FREESOUND_API_KEY")
        self.attr_file = self.target_dir / "ATTRIBUTIONS.txt"

    def log_attribution(self, filename: str, category: str, license_str: str, url: str, description: str):
        """Records provenance and attribution for an ingested file."""
        line = f"{filename} | {category} | License: {license_str} | URL: {url} | {description}\n"
        with open(self.attr_file, "a", encoding="utf-8") as f:
            f.write(line)

    def download_file(self, url: str, destination: Path, chunk_size: int = 1024 * 64, max_bytes: Optional[int] = None) -> bool:
        """Streams a download with progress tracking, with optional max_bytes cap."""
        try:
            headers = {"User-Agent": "RainAI-Dataset-Collector/1.0"}
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            if max_bytes and total_size > max_bytes:
                total_size = max_bytes
                
            downloaded = 0
            with open(destination, "wb") as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
                        downloaded += len(chunk)
                        if max_bytes and downloaded >= max_bytes:
                            break
            return True
        except Exception as e:
            print(f"[!] Download failed for {url}: {e}")
            if destination.exists():
                destination.unlink()
            return False

    def ingest_curated_sources(self) -> List[Path]:
        """Ingests all pre-vetted CC0 / CC-BY / Public Domain curated audio items."""
        downloaded = []
        print(f"[*] Ingesting {len(CURATED_SOURCES)} curated open datasets (surfaces, RIRs, rain, thunder)...")
        for item in CURATED_SOURCES:
            valid, reason = LicenseVerifier.verify(item.license)
            if not valid:
                print(f"[-] Skipping curated item {item.target_filename}: {reason}")
                continue

            dest = (self.rirs_dir if "rir" in item.category else self.target_dir) / item.target_filename
            if dest.exists():
                print(f"[+] Already downloaded: {dest.name}")
                downloaded.append(dest)
                continue
                
            print(f"[+] Downloading {dest.name} [{item.license}]...")
            if self.download_file(item.url, dest, max_bytes=30 * 1024 * 1024):  # 30MB max per item
                downloaded.append(dest)
                self.log_attribution(item.target_filename, item.category, item.license, item.url, item.description)
                    
        return downloaded

    def ingest_bigsoundbank(
        self,
        queries: List[str] = ["rain", "thunder", "pluie", "orage"],
        max_items_per_query: int = 4
    ) -> List[Path]:
        """
        Scrapes and downloads CC0 sound recordings directly from BigSoundBank.
        All BigSoundBank media is strictly CC0 (Public Domain).
        """
        downloaded = []
        base_url = "https://bigsoundbank.com"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RainAI/1.0"}

        for q in queries:
            print(f"[*] Querying BigSoundBank for '{q}'...")
            search_url = f"{base_url}/search"
            try:
                resp = requests.get(search_url, params={"q": q}, headers=headers, timeout=20)
                if resp.status_code != 200:
                    continue
                
                items_to_process = []
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    audio_tags = soup.find_all("audio")
                    for audio in audio_tags:
                        source = audio.find("source")
                        if source and source.get("src"):
                            rel_path = source["src"]
                            card = audio.find_parent("div")
                            desc = "BigSoundBank sound effect"
                            if card:
                                outer_card = card.find_parent("div")
                                if outer_card:
                                    text_parts = outer_card.get_text(separator=" | ", strip=True).split(" | ")
                                    desc = " - ".join(text_parts[:3])
                            items_to_process.append((rel_path, desc))
                else:
                    # Regex fallback
                    raw_sources = re.findall(r'<source[^>]+src=[\x27\x22](/UPLOAD/[^\x27\x22]+)[\x27\x22]', resp.text)
                    for rel_path in raw_sources:
                        items_to_process.append((rel_path, f"BigSoundBank sound effect ({q})"))

                count = 0
                for rel_audio_path, desc in items_to_process:
                    if count >= max_items_per_query:
                        break

                    sound_id = Path(rel_audio_path).stem
                    full_audio_url = f"{base_url}{rel_audio_path}"

                    filename = f"bigsoundbank_{sound_id}_{q}.mp3"
                    dest_path = self.target_dir / filename

                    if dest_path.exists():
                        print(f"[+] Already downloaded: {dest_path.name}")
                        downloaded.append(dest_path)
                        count += 1
                        continue

                    print(f"[+] Downloading BigSoundBank item: {filename} [CC0]...")
                    if self.download_file(full_audio_url, dest_path, max_bytes=25 * 1024 * 1024):
                        downloaded.append(dest_path)
                        self.log_attribution(filename, f"bigsoundbank_{q}", "CC0", full_audio_url, desc)
                        count += 1

            except Exception as e:
                print(f"[!] Error querying BigSoundBank for '{q}': {e}")

        return downloaded

    def ingest_figshare_rainfall(self, article_id: str = "3156082") -> List[Path]:
        """
        Retrieves the Figshare Freesound1010 rainfall research dataset metadata and files.
        (DOI: 10.6084/m9.figshare.3156082, License: CC-BY 4.0)
        """
        downloaded = []
        api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        headers = {"User-Agent": "RainAI-Figshare-Ingester/1.0"}

        print(f"[*] Querying Figshare API for rainfall dataset (ID: {article_id})...")
        try:
            resp = requests.get(api_url, headers=headers, timeout=20)
            if resp.status_code != 200:
                print(f"[!] Figshare returned status {resp.status_code}")
                return []

            data = resp.json()
            title = data.get("title", "Freesound1010 Rainfall Dataset")
            license_data = data.get("license", {})
            license_name = license_data.get("name", "CC-BY 4.0")

            valid, reason = LicenseVerifier.verify(license_name)
            if not valid:
                print(f"[-] Figshare dataset rejected by license verifier: {reason}")
                return []

            files = data.get("files", [])
            for f_info in files:
                f_name = f_info.get("name")
                f_url = f_info.get("download_url")
                if not f_name or not f_url:
                    continue

                target_dest = self.target_dir / f_name
                if target_dest.exists():
                    print(f"[+] Already downloaded: {target_dest.name}")
                    downloaded.append(target_dest)
                    continue

                print(f"[+] Downloading Figshare file: {f_name} [{license_name}]...")
                if self.download_file(f_url, target_dest, max_bytes=50 * 1024 * 1024):
                    downloaded.append(target_dest)
                    self.log_attribution(f_name, "rainfall_benchmark", license_name, f_url, title)

        except Exception as e:
            print(f"[!] Figshare ingestion error: {e}")

        return downloaded

    def query_freesound(
        self,
        queries: Optional[List[str]] = None,
        page_size: int = 10
    ) -> List[Path]:
        """
        Queries the Freesound API across a matrix of rain and atmospheric conditions,
        strictly filtering for CC0 / CC-BY licenses.
        """
        if not self.freesound_api_key:
            print("[!] Freesound API key not found. Set FREESOUND_API_KEY to search Freesound.")
            return []

        if queries is None:
            queries = [
                "heavy torrential rain", "gentle drizzle", "rain on tin roof",
                "rain on window glass", "rain in forest", "thunderclap storm",
                "rain drops puddle", "rain on canvas tent"
            ]

        downloaded = []
        url = "https://freesound.org/apiv2/search/text/"

        for query in queries:
            params = {
                "query": query,
                "token": self.freesound_api_key,
                "filter": "duration:[5.0 TO 300.0] license:\"Creative Commons 0\"",
                "fields": "id,name,previews,duration,license,description",
                "page_size": page_size
            }
            
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                
                for item in results:
                    sound_id = item.get("id")
                    name = item.get("name", "sound")
                    license_str = item.get("license", "CC0")

                    valid, _ = LicenseVerifier.verify(license_str)
                    if not valid:
                        continue

                    previews = item.get("previews", {})
                    preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                    if not preview_url:
                        continue

                    safe_title = "".join(c for c in name if c.isalnum() or c in " ._-").strip()
                    dest_file = self.target_dir / f"freesound_{sound_id}_{safe_title}.mp3"

                    if dest_file.exists():
                        downloaded.append(dest_file)
                        continue

                    print(f"[+] Downloading Freesound clip: {dest_file.name}...")
                    if self.download_file(preview_url, dest_file):
                        downloaded.append(dest_file)
                        self.log_attribution(
                            dest_file.name,
                            f"freesound_{query.replace(' ', '_')}",
                            license_str,
                            preview_url,
                            item.get("description", "")[:120]
                        )

            except Exception as e:
                print(f"[!] Freesound search failed for query '{query}': {e}")

        return downloaded

    def ingest_wikimedia_commons(
        self, 
        queries: Optional[List[str]] = None,
        max_files_per_query: int = 6
    ) -> List[Path]:
        """
        Queries Wikimedia Commons for audio files, verifies compatible open licensing 
        (CC0, Public Domain, CC-BY, CC-BY-SA, strictly rejecting NonCommercial NC),
        and downloads the audio with attribution records.
        """
        if queries is None:
            queries = [
                "rain filetype:audio",
                "thunderstorm filetype:audio",
                "forest rain filetype:audio",
                "rain on roof filetype:audio",
                "rain storm filetype:audio",
                "drizzle filetype:audio",
                "forest wind filetype:audio"
            ]

        downloaded = []
        headers = {"User-Agent": "RainAI-AudioIngester/1.0 (contact@rainai.local)"}
        api_url = "https://commons.wikimedia.org/w/api.php"
        
        for q in queries:
            print(f"[*] Searching Wikimedia Commons for '{q}'...")
            params = {
                "action": "query",
                "list": "search",
                "srsearch": q,
                "srnamespace": 6,
                "srlimit": max_files_per_query,
                "format": "json"
            }
            try:
                res = requests.get(api_url, headers=headers, params=params, timeout=15).json()
                search_results = res.get("query", {}).get("search", [])
            except Exception as e:
                print(f"[!] Search query failed: {e}")
                continue

            for item in search_results:
                title = item["title"]
                info_params = {
                    "action": "query",
                    "titles": title,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "format": "json"
                }
                try:
                    info_res = requests.get(api_url, headers=headers, params=info_params, timeout=15).json()
                    pages = info_res.get("query", {}).get("pages", {})
                    page = list(pages.values())[0]
                    img_info = page.get("imageinfo", [{}])[0]
                    file_url = img_info.get("url")
                    if not file_url:
                        continue
                        
                    extmeta = img_info.get("extmetadata", {})
                    license_name = extmeta.get("LicenseShortName", {}).get("value", "Unknown")
                    
                    # STRICT LICENSE FILTER: Must be compatible with commercial/open use
                    is_valid, reason = LicenseVerifier.verify(license_name)
                    if not is_valid:
                        print(f"[-] Skipping {title}: {reason}")
                        continue
                        
                    # Target filename sanitized
                    safe_name = "".join(c for c in title.replace("File:", "") if c.isalnum() or c in " ._-").strip()
                    dest_file = self.target_dir / safe_name
                    
                    clean_name = safe_name.encode('ascii', 'ignore').decode()
                    if dest_file.exists():
                        print(f"[+] Already exists: {clean_name}")
                        downloaded.append(dest_file)
                        continue
                        
                    print(f"[+] Downloading {clean_name} [{license_name}]...")
                    if self.download_file(file_url, dest_file):
                        downloaded.append(dest_file)
                        self.log_attribution(safe_name, "wikimedia_commons", license_name, file_url, title)
                except Exception as e:
                    print(f"[!] Error fetching audio item: {e}")

        return downloaded

    def ingest_all(self) -> Dict[str, List[Path]]:
        """Orchestrates ingestion across all integrated open repositories."""
        results = {
            "curated": self.ingest_curated_sources(),
            "bigsoundbank": self.ingest_bigsoundbank(),
            "figshare": self.ingest_figshare_rainfall(),
            "wikimedia": self.ingest_wikimedia_commons()
        }
        if self.freesound_api_key:
            results["freesound"] = self.query_freesound()
            
        print(f"[✓] Ingestion complete. Attribution log maintained at: {self.attr_file}")
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RainAI Multi-Source Audio Ingest Pipeline")
    parser.add_argument("--target-dir", type=str, default="./data/raw_audio", help="Destination audio directory")
    parser.add_argument("--sources", nargs="+", default=["curated", "bigsoundbank", "figshare", "wikimedia"], help="Sources to ingest")
    args = parser.parse_args()

    ingester = RainAudioIngester(target_dir=Path(args.target_dir))
    if "curated" in args.sources:
        ingester.ingest_curated_sources()
    if "bigsoundbank" in args.sources:
        ingester.ingest_bigsoundbank()
    if "figshare" in args.sources:
        ingester.ingest_figshare_rainfall()
    if "wikimedia" in args.sources:
        ingester.ingest_wikimedia_commons()
