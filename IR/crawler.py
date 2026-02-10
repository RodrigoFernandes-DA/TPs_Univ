import requests
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import time
from typing import Set, Optional


class WebCrawler:
    def __init__(self, max_depth: int = 3, delay: float = 0.1):
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; WebCrawler/1.0)'
        })
        
    def _should_ignore_url(self, url: str, base_domain: str) -> bool:
        parsed_url = urlparse(url)
        
        # Skip if not HTTP/HTTPS
        if parsed_url.scheme not in ('http', 'https'):
            return True
            
        # Skip if same domain check fails (for Wikipedia, we want to stay on Wikipedia)
        if base_domain not in parsed_url.netloc:
            return True
            
        # Skip common non-content pages
        ignore_paths = [
            '/wiki/Special:', '/wiki/Talk:', '/wiki/User:', '/wiki/User_talk:',
            '/wiki/Wikipedia:', '/wiki/Wikipedia_talk:', '/wiki/File:', '/wiki/MediaWiki:',
            '/wiki/Template:', '/wiki/Template_talk:', '/wiki/Help:', '/wiki/Help_talk:',
            '/wiki/Category:', '/wiki/Category_talk:', '/wiki/Portal:', '/wiki/Portal_talk:',
            '/wiki/Draft:', '/wiki/Draft_talk:', '/wiki/TimedText:', '/wiki/Module:',
            '/wiki/Module_talk:', '/wiki/Media:', '/wiki/Thread:', '/wiki/Summary:',
            '/wiki/Index:', '/wiki/Book:', '/wiki/Main_Page', '/wiki/Aide', '/wiki/Cat%C3%A9gorie',
            '/wiki/Sp%C3%A9cial', '/wiki/Wikip%C3%A9dia', '/wiki/Portail:'
        ]

        # For Wikipedia, skip administrative and non-article pages
        if any(url_path in parsed_url.path for url_path in ignore_paths):
            return True
            
        # Skip URLs with certain fragments or query parameters
        if parsed_url.fragment:
            # Skip links that point to specific sections (can cause duplicates)
            if parsed_url.fragment.startswith('cite_note'):
                return True
                
        # Skip certain file extensions
        ignore_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.svg', 
                           '.zip', '.tar', '.gz', '.mp3', '.mp4', '.avi']
        if any(parsed_url.path.lower().endswith(ext) for ext in ignore_extensions):
            return True
            
        return False
        
    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        
        # Remove fragments (after #)
        parsed = parsed._replace(fragment='')
        
        # Normalize scheme and netloc to lowercase
        parsed = parsed._replace(scheme=parsed.scheme.lower(),
                                netloc=parsed.netloc.lower())
        
        # For Wikipedia, remove query parameters that don't affect content
        if 'wikipedia.org' in parsed.netloc:
            # Keep only specific query parameters that might be relevant
            # For most Wikipedia links, we can ignore all query parameters
            parsed = parsed._replace(query='')
            
        return urlunparse(parsed)
        
    def _get_html(self, url: str) -> Optional[str]:
        try:
            time.sleep(self.delay)  # Be polite
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if it's HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return None
                
            return response.text
            
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching {url}: {e}")
            return None
            
    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        if not html:
            return set()
            
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            
            # Normalize the URL
            normalized_url = self._normalize_url(absolute_url)
            
            links.add(normalized_url)
            
        return links
        
    def crawl(self, start_url: str, depth: Optional[int] = None) -> Set[str]:
        if depth is None:
            depth = self.max_depth
            
        # Reset visited URLs for new crawl
        self.visited_urls = set()
        
        # Normalize start URL
        start_url = self._normalize_url(start_url)
        
        # Get base domain for filtering
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc
        
        print(f"Starting crawl from: {start_url}")
        print(f"Maximum depth: {depth}")
        print(f"Base domain: {base_domain}")
        print("-" * 50)
        
        # Start recursive crawling
        self._recursive_crawl(start_url, depth, base_domain)
        
        print("-" * 50)
        print(f"Crawling completed!")
        print(f"Total unique URLs discovered: {len(self.visited_urls)}")
        
        return self.visited_urls
        
    def _recursive_crawl(self, url: str, current_depth: int, base_domain: str):
        # Base cases
        if current_depth < 0:
            return
            
        if url in self.visited_urls:
            return
            
        # Check if URL should be ignored
        if self._should_ignore_url(url, base_domain):
            return
            
        # Mark as visited
        self.visited_urls.add(url)
        print(f"Depth {self.max_depth - current_depth}: {url}")
        
        # Get HTML content
        html = self._get_html(url)
        if not html:
            return
            
        # Extract links from HTML
        links = self._extract_links(html, url)
        
        # Recursively crawl links
        for link in links:
            self._recursive_crawl(link, current_depth - 1, base_domain)


def crawl_wikipedia(start_url: str, recursion_depth: int) -> Set[str]:
    crawler = WebCrawler(max_depth=recursion_depth, delay=0.2) # create
    discovered_urls = crawler.crawl(start_url) # crawl
    
    return discovered_urls


if __name__ == "__main__":
    test_url = "https://fr.wikipedia.org/wiki/L%27%C3%89vangile_du_monstre_en_spaghettis_volant"
    depth = 2
    
    print("Testing Wikipedia crawler...")
    print("=" * 50)
    
    urls = crawl_wikipedia(test_url, depth)
        
    print(f"\nTotal URLs discovered: {len(urls)}")

    