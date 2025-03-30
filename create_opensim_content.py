import os
import requests
import random
import time
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import re

# Create output directory
output_dir = "./opensim_content"
os.makedirs(output_dir, exist_ok=True)

# Headers for HTTP requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def sanitize_filename(text):
    """Create a valid filename from text"""
    return re.sub(r'[\\/*?:"<>|]', "_", text)

# Function to scrape SimTK forum posts
def scrape_simtk_forums(num_pages=20):
    base_url = "https://simtk.org/plugins/phpBB/indexPhpbb.php?group_id=91&pluginname=phpBB"
    forum_content = []
    
    print(f"Scraping SimTK forums (up to {num_pages} pages)...")
    
    # First get forum sections
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find forum links
    forum_links = []
    for a in soup.find_all('a', href=True):
        if 'viewforum.php' in a['href']:
            forum_links.append(a['href'])
    
    # Process each forum
    for forum_link in forum_links[:5]:  # Limit to first 5 forums
        forum_url = f"https://simtk.org{forum_link}"
        
        # Get threads from forum
        for page in range(1, num_pages // 5 + 1):
            try:
                page_url = f"{forum_url}&start={(page-1)*30}"
                response = requests.get(page_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find thread links
                thread_links = []
                for a in soup.find_all('a', href=True):
                    if 'viewtopic.php' in a['href'] and 'sid=' not in a['href']:
                        thread_links.append(a['href'])
                
                # Remove duplicates
                thread_links = list(set(thread_links))
                
                # Process a sample of threads from this page
                for thread_link in thread_links[:5]:  # Limit to 5 threads per page
                    thread_url = f"https://simtk.org{thread_link}"
                    
                    try:
                        response = requests.get(thread_url, headers=headers)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Get thread title
                        title_elem = soup.find('h2')
                        title = title_elem.text.strip() if title_elem else "Unknown Thread"
                        
                        # Get post content
                        posts = []
                        for post_div in soup.find_all('div', class_='post'):
                            # Get post author
                            author_elem = post_div.find('p', class_='author')
                            author = author_elem.text.strip() if author_elem else "Unknown Author"
                            
                            # Get post content
                            content_div = post_div.find('div', class_='content')
                            content = content_div.text.strip() if content_div else ""
                            
                            if content:
                                posts.append({
                                    "author": author,
                                    "content": content
                                })
                        
                        # Save thread data
                        thread_data = {
                            "title": title,
                            "url": thread_url,
                            "posts": posts
                        }
                        
                        forum_content.append(thread_data)
                        
                        # Save individual thread file
                        filename = f"forum_{sanitize_filename(title)[:80]}.txt"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"TITLE: {title}\n")
                            f.write(f"URL: {thread_url}\n\n")
                            
                            for post in posts:
                                f.write(f"AUTHOR: {post['author']}\n")
                                f.write(f"CONTENT:\n{post['content']}\n\n")
                                f.write("-" * 80 + "\n\n")
                        
                        print(f"Saved forum thread: {title}")
                        
                    except Exception as e:
                        print(f"Error processing thread {thread_url}: {e}")
                    
                    # Short delay between thread requests
                    time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                print(f"Error processing forum page {page_url}: {e}")
            
            # Delay between page requests
            time.sleep(random.uniform(2, 3))
    
    return forum_content

# Function to scrape GitHub repository contents
def scrape_github_repo(repo="opensim-org/opensim-core", max_files=50):
    api_url = f"https://api.github.com/repos/{repo}/contents"
    github_content = []
    
    print(f"Scraping GitHub repository: {repo}...")
    
    def process_dir(path, depth=0, file_count=0):
        nonlocal github_content
        
        if depth > 3 or file_count >= max_files:  # Limit depth and files
            return file_count
        
        try:
            response = requests.get(f"{api_url}/{path}" if path else api_url, headers=headers)
            if response.status_code != 200:
                return file_count
            
            items = response.json()
            
            for item in items:
                if file_count >= max_files:
                    return file_count
                
                if item['type'] == 'file':
                    # Check if it's a text file we're interested in
                    if any(item['name'].endswith(ext) for ext in ['.md', '.txt', '.cpp', '.h', '.py', '.xml']):
                        try:
                            # Get file content
                            file_response = requests.get(item['download_url'], headers=headers)
                            if file_response.status_code == 200:
                                file_content = file_response.text
                                
                                # Save file
                                filename = f"github_{sanitize_filename(item['path'])}"
                                filepath = os.path.join(output_dir, filename)
                                
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write(file_content)
                                
                                github_content.append({
                                    "name": item['name'],
                                    "path": item['path'],
                                    "url": item['html_url']
                                })
                                
                                print(f"Saved GitHub file: {item['path']}")
                                file_count += 1
                                
                                # Short delay between file requests
                                time.sleep(random.uniform(0.5, 1))
                        except Exception as e:
                            print(f"Error downloading file {item['path']}: {e}")
                
                elif item['type'] == 'dir' and depth < 3:
                    # Process subdirectory recursively
                    file_count = process_dir(item['path'], depth + 1, file_count)
        
        except Exception as e:
            print(f"Error processing directory {path}: {e}")
        
        return file_count
    
    process_dir("", 0, 0)
    return github_content

# Function to download documentation from OpenSim website
def download_opensim_docs():
    docs_urls = [
        "https://simtk-confluence.stanford.edu:8443/display/OpenSim/OpenSim+Documentation",
        "https://simtk-confluence.stanford.edu:8443/display/OpenSim/User%27s+Guide",
        "https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python",
        "https://simtk-confluence.stanford.edu:8443/display/OpenSim/Examples+and+Tutorials"
    ]
    
    docs_content = []
    
    print("Downloading OpenSim documentation...")
    
    for url in docs_urls:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to access {url}: HTTP {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page title
            title_elem = soup.find('h1', id='title-text')
            title = title_elem.text.strip() if title_elem else "Unknown Page"
            
            # Extract main content
            content_div = soup.find('div', id='main-content')
            content = content_div.text.strip() if content_div else ""
            
            if content:
                docs_content.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
                
                # Save individual page file
                filename = f"doc_{sanitize_filename(title)}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"TITLE: {title}\n")
                    f.write(f"URL: {url}\n\n")
                    f.write(content)
                
                print(f"Saved documentation page: {title}")
                
                # Get linked pages
                for a in content_div.find_all('a', href=True):
                    if a['href'].startswith('/display/OpenSim/'):
                        linked_url = f"https://simtk-confluence.stanford.edu:8443{a['href']}"
                        
                        try:
                            sub_response = requests.get(linked_url, headers=headers)
                            if sub_response.status_code != 200:
                                continue
                            
                            sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                            
                            # Extract page title
                            sub_title_elem = sub_soup.find('h1', id='title-text')
                            sub_title = sub_title_elem.text.strip() if sub_title_elem else "Unknown Subpage"
                            
                            # Extract main content
                            sub_content_div = sub_soup.find('div', id='main-content')
                            sub_content = sub_content_div.text.strip() if sub_content_div else ""
                            
                            if sub_content:
                                docs_content.append({
                                    "title": sub_title,
                                    "url": linked_url,
                                    "content": sub_content
                                })
                                
                                # Save individual page file
                                sub_filename = f"doc_{sanitize_filename(sub_title)}.txt"
                                sub_filepath = os.path.join(output_dir, sub_filename)
                                
                                with open(sub_filepath, 'w', encoding='utf-8') as f:
                                    f.write(f"TITLE: {sub_title}\n")
                                    f.write(f"URL: {linked_url}\n\n")
                                    f.write(sub_content)
                                
                                print(f"Saved linked documentation page: {sub_title}")
                            
                            # Delay between subpage requests
                            time.sleep(random.uniform(1, 2))
                            
                        except Exception as e:
                            print(f"Error processing linked page {linked_url}: {e}")
            
            # Delay between page requests
            time.sleep(random.uniform(2, 3))
            
        except Exception as e:
            print(f"Error processing documentation page {url}: {e}")
    
    return docs_content

# Run the scrapers
forum_content = scrape_simtk_forums(num_pages=30)
github_content = scrape_github_repo(max_files=50)
docs_content = download_opensim_docs()

# Save overall report
report = {
    "forums": {
        "count": len(forum_content),
        "items": forum_content
    },
    "github": {
        "count": len(github_content),
        "items": github_content
    },
    "docs": {
        "count": len(docs_content),
        "items": docs_content
    }
}

report_path = os.path.join(output_dir, "content_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print(f"\nScraped content summary:")
print(f"Forum threads: {len(forum_content)}")
print(f"GitHub files: {len(github_content)}")
print(f"Documentation pages: {len(docs_content)}")
print(f"Total items: {len(forum_content) + len(github_content) + len(docs_content)}")
print(f"All content saved to: {output_dir}")