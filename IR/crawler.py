import requests
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import time
from typing import Set, Optional
from collections import deque


class WebCrawler:
    def __init__(self, max_depth: int = 3, delay: float = 0.1, max_pages: int = 200):
        self.max_depth = max_depth
        self.delay = delay
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; WebCrawler/1.0)'
        })

    def _should_ignore_url(self, url: str, base_domain: str) -> bool:
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ('http', 'https'):
            return True

        if base_domain not in parsed_url.netloc:
            return True

        ignore_paths = [ '/wiki/Special:', '/wiki/Talk:', '/wiki/User:', '/wiki/User_talk:', '/wiki/Wikipedia:', '/wiki/Wikipedia_talk:', '/wiki/File:', '/wiki/MediaWiki:', '/wiki/Template:', '/wiki/Template_talk:', '/wiki/Help:', '/wiki/Help_talk:', '/wiki/Category:', '/wiki/Category_talk:', '/wiki/Portal:', '/wiki/Portal_talk:', '/wiki/Draft:', '/wiki/Draft_talk:', '/wiki/TimedText:', '/wiki/Module:', '/wiki/Module_talk:', '/wiki/Media:', '/wiki/Thread:', '/wiki/Summary:', '/wiki/Index:', '/wiki/Book:', '/wiki/Main_Page', '/wiki/Aide', '/wiki/Cat%C3%A9gorie', '/wiki/Sp%C3%A9cial', '/wiki/Wikip%C3%A9dia', '/wiki/Portail:', '/wiki/Wikipédia:Contact', '/wiki/Anglais',
                         '/wiki/Mod%C3%A8le:', 'wiki/Projet:', '/wiki/Wikipédia:', '/wiki/Wikisource', 'wiki/Discussion:', '/wiki/18', '/wiki/19' , '/wiki/20', 'wiki/Wiki', '/w/index.php', '/wiki/Discussion_Wikip%C3%A9dia:Contact', '/Wikipédia:Contact', '/wiki/Mod%C3%A8le:Infobox_Universit%C3%A9'  ]

        if any(p in parsed_url.path for p in ignore_paths):
            return True

        ignore_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.svg',
                             '.zip', '.tar', '.gz', '.mp3', '.mp4', '.avi']
        if any(parsed_url.path.lower().endswith(ext) for ext in ignore_extensions):
            return True

        return False

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        parsed = parsed._replace(fragment='')
        parsed = parsed._replace(scheme=parsed.scheme.lower(),
                                 netloc=parsed.netloc.lower())

        if 'wikipedia.org' in parsed.netloc:
            parsed = parsed._replace(query='')

        return urlunparse(parsed)

    def _get_html(self, url: str) -> Optional[str]:
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            if 'text/html' not in response.headers.get('content-type', '').lower():
                return None

            return response.text

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        soup = BeautifulSoup(html, 'html.parser')
        links = set()

        for a in soup.find_all('a', href=True):
            absolute = urljoin(base_url, a['href'])
            links.add(self._normalize_url(absolute))

        return links

    def crawl(self, start_url: str) -> Set[str]:
        start_url = self._normalize_url(start_url)
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc

        print(f"Starting BFS crawl from: {start_url}")
        print(f"Max depth: {self.max_depth}")
        print(f"Max pages: {self.max_pages}")
        print("-" * 60)

        queue = deque()
        queue.append((start_url, 0))

        while queue and len(self.visited_urls) < self.max_pages:
            url, depth = queue.popleft()

            if depth > self.max_depth:
                continue

            if url in self.visited_urls:
                continue

            if self._should_ignore_url(url, base_domain):
                continue

            self.visited_urls.add(url)
            print(f"[{len(self.visited_urls):03}] Depth {depth}: {url}")

            html = self._get_html(url)
            if not html:
                continue

            for link in self._extract_links(html, url):
                if link not in self.visited_urls:
                    queue.append((link, depth + 1))

        print("-" * 60)
        print(f"Crawl finished — pages crawled: {len(self.visited_urls)}")

        return self.visited_urls


def crawl_wikipedia(start_url: str, depth: int, max_pages: int) -> Set[str]:
    crawler = WebCrawler(max_depth=depth, delay=0.2, max_pages=max_pages)
    return crawler.crawl(start_url)


if __name__ == "__main__":
    test_url = "https://fr.wikipedia.org/wiki/L%27%C3%89vangile_du_monstre_en_spaghettis_volant"
    depth = 2
    max_pages = 100

    urls = crawl_wikipedia(test_url, depth, max_pages)
    print(f"\nTotal URLs discovered: {len(urls)}")
