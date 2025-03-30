import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import json

class OpenSimCrawler:
    def __init__(self, start_urls, output_dir="./scraped_data", delay=1):
        """
        Initialize the OpenSim documentation crawler.
        
        Args:
            start_urls (list): List of URLs to start crawling from
            output_dir (str): Directory to save scraped data
            delay (int): Delay between requests in seconds to respect server load
        """
        self.start_urls = start_urls
        self.output_dir = output_dir
        self.delay = delay
        self.visited_urls = set()
        self.document_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    
    def is_valid_url(self, url):
        """
        Check if a URL is valid for crawling based on domain and file type.
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if URL is valid for crawling, False otherwise
        """
        # Parse the URL to extract the domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Define allowed domains (OpenSim documentation sources)
        allowed_domains = [
            'simtk.org',
            'simtk-confluence.stanford.edu',
            'opensim.stanford.edu'
        ]
        
        # Check if the domain is in the allowed list
        domain_valid = any(domain.endswith(allowed_domain) for allowed_domain in allowed_domains)
        
        # Check if the URL path doesn't end with file extensions we want to skip
        path = parsed_url.path.lower()
        invalid_extensions = ['.pdf', '.zip', '.exe', '.tar.gz', '.jpg', '.jpeg', '.png', '.gif', '.svg']
        path_valid = not any(path.endswith(ext) for ext in invalid_extensions)
        
        return domain_valid and path_valid
    
    def extract_text_from_html(self, html_content, url):
        """
        Extract clean text content from HTML.
        
        Args:
            html_content (str): HTML content to parse
            url (str): Source URL for reference
            
        Returns:
            dict: Document with extracted content and metadata
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to extract the title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.strip()
        
        # Try to extract the main content based on common content containers
        # This will need customization based on the structure of OpenSim documentation
        content_containers = [
            soup.find('div', {'id': 'main-content'}),  # Common in Confluence
            soup.find('div', {'class': 'wiki-content'}),  # Common in Confluence
            soup.find('div', {'id': 'content'}),
            soup.find('article'),
            soup.find('main'),
        ]
        
        # Use the first non-None container or fall back to body
        main_content = next((container for container in content_containers if container), soup.body)
        
        # Remove script, style, and navigation elements
        for element in main_content.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Extract the text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Create document structure
        document = {
            "id": f"doc_{self.document_count}",
            "url": url,
            "title": title,
            "content_type": self.determine_content_type(url, title),
            "category": self.determine_category(url, title),
            "hierarchy": self.extract_hierarchy(url, soup),
            "difficulty": self.determine_difficulty(url, title, text),
            "raw_text": text,
            "extracted_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return document
    
    def determine_content_type(self, url, title):
        """
        Determine the content type based on URL and title.
        
        Args:
            url (str): Source URL
            title (str): Document title
            
        Returns:
            str: Content type (tutorial, reference, api, example, forum)
        """
        url_lower = url.lower()
        title_lower = title.lower()
        
        if 'api_docs' in url_lower or 'api' in url_lower:
            return "api"
        elif 'tutorial' in url_lower or 'tutorial' in title_lower:
            return "tutorial"
        elif 'example' in url_lower or 'example' in title_lower:
            return "example"
        elif 'forum' in url_lower or 'forum' in title_lower:
            return "forum"
        else:
            return "reference"
    
    def determine_category(self, url, title):
        """
        Determine the category based on URL and title.
        
        Args:
            url (str): Source URL
            title (str): Document title
            
        Returns:
            str: Category
        """
        url_lower = url.lower()
        title_lower = title.lower()
        
        if 'getting_started' in url_lower or 'getting started' in title_lower:
            return "getting_started"
        elif 'modeling' in url_lower or 'modeling' in title_lower:
            return "modeling"
        elif 'analysis' in url_lower or 'analysis' in title_lower:
            return "analysis"
        elif 'simulation' in url_lower or 'simulation' in title_lower:
            return "simulation"
        else:
            return "general"
    
    def extract_hierarchy(self, url, soup):
        """
        Extract document hierarchy from URL or breadcrumbs.
        
        Args:
            url (str): Source URL
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            list: Hierarchy path
        """
        # Try to extract from breadcrumbs (common in documentation sites)
        breadcrumbs = soup.find('ul', {'class': 'breadcrumb'}) or soup.find('nav', {'class': 'breadcrumb'})
        
        if breadcrumbs:
            hierarchy = [item.text.strip() for item in breadcrumbs.find_all('li') if item.text.strip()]
            return hierarchy
        
        # If breadcrumbs not found, extract from URL
        path_parts = urlparse(url).path.strip('/').split('/')
        # Filter out empty parts and common elements like 'display'
        hierarchy = [part for part in path_parts if part and part != 'display']
        
        return hierarchy if hierarchy else ["Root"]
    
    def determine_difficulty(self, url, title, content):
        """
        Estimate content difficulty level.
        
        Args:
            url (str): Source URL
            title (str): Document title
            content (str): Document content
            
        Returns:
            str: Difficulty level
        """
        url_lower = url.lower()
        title_lower = title.lower()
        
        if 'beginner' in url_lower or 'beginner' in title_lower or 'getting started' in title_lower:
            return "beginner"
        elif 'advanced' in url_lower or 'advanced' in title_lower:
            return "advanced"
        else:
            return "intermediate"
    
    def extract_links(self, soup, base_url):
        """
        Extract all links from the page.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            base_url (str): Base URL for resolving relative links
            
        Returns:
            list: List of absolute URLs
        """
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            absolute_url = urljoin(base_url, href)
            
            # Skip anchor links within the same page
            if absolute_url == base_url or absolute_url.startswith(base_url + '#'):
                continue
                
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
        
        return links
    
    def crawl(self, max_pages=100, max_depth=3):
        """
        Crawl OpenSim documentation pages starting from start_urls.
        
        Args:
            max_pages (int): Maximum number of pages to crawl
            max_depth (int): Maximum crawl depth
        """
        queue = [(url, 0) for url in self.start_urls]  # (url, depth)
        
        with tqdm(total=max_pages, desc="Crawling") as pbar:
            while queue and len(self.visited_urls) < max_pages:
                url, depth = queue.pop(0)
                
                # Skip if already visited or depth exceeded
                if url in self.visited_urls or depth > max_depth:
                    continue
                
                self.visited_urls.add(url)
                
                try:
                    # Respect robots.txt by adding delay
                    time.sleep(self.delay)
                    
                    # Fetch the page
                    response = requests.get(url, headers=self.headers, timeout=10)
                    
                    # Skip non-HTML responses
                    if 'text/html' not in response.headers.get('Content-Type', ''):
                        pbar.update(1)
                        continue
                    
                    # Parse the page
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract and save document
                    document = self.extract_text_from_html(response.text, url)
                    self.save_document(document)
                    
                    # Extract links for next level
                    if depth < max_depth:
                        links = self.extract_links(soup, url)
                        for link in links:
                            if link not in self.visited_urls:
                                queue.append((link, depth + 1))
                    
                    self.document_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
                    pbar.update(1)
        
        print(f"Crawling complete. Scraped {self.document_count} documents.")
    
    def save_document(self, document):
        """
        Save document to JSON file.
        
        Args:
            document (dict): Document to save
        """
        filename = os.path.join(self.output_dir, f"{document['id']}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Define the starting URLs
    start_urls = [
        "https://simtk.org/projects/opensim",
        "https://simtk-confluence.stanford.edu/display/OpenSim/User%27s+Guide",
        "https://simtk-confluence.stanford.edu/display/OpenSim/Tutorials",
        "https://simtk.org/api_docs/opensim/api_docs/"
    ]
    
    # Create and run the crawler
    crawler = OpenSimCrawler(start_urls=start_urls, delay=2)
    crawler.crawl(max_pages=50, max_depth=2)  # Start with a limited crawl for testing