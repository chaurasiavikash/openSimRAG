import os
import json
import re
from tqdm import tqdm
import html
from bs4 import BeautifulSoup
import glob

class ContentProcessor:
    def __init__(self, input_dir="./scraped_data", output_dir="./processed_data"):
        """
        Initialize the content processor.
        
        Args:
            input_dir (str): Directory containing scraped data
            output_dir (str): Directory to save processed data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_text(self, text):
        """
        Clean and normalize text content.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Decode HTML entities
        text = html.unescape(text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_code_blocks(self, html_content):
        """
        Extract code blocks from HTML content.
        
        Args:
            html_content (str): HTML content
            
        Returns:
            list: List of extracted code blocks with metadata
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        code_blocks = []
        
        # Find all code elements
        for idx, code_elem in enumerate(soup.find_all(['code', 'pre'])):
            language = "unknown"
            
            # Try to determine language from class
            if code_elem.has_attr('class'):
                class_str = ' '.join(code_elem['class'])
                if 'language-' in class_str:
                    language_match = re.search(r'language-(\w+)', class_str)
                    if language_match:
                        language = language_match.group(1)
            
            # Get the code content
            code_content = code_elem.get_text(strip=True)
            
            code_blocks.append({
                "id": idx,
                "language": language,
                "content": code_content
            })
        
        return code_blocks
    
    def preserve_special_formatting(self, text):
        """
        Preserve special formatting in text like equations and lists.
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Processed text with preserved formatting
        """
        # Preserve equations (simple regex for common formats like $...$ or \(...\))
        text = re.sub(r'\$([^$]+)\$', r'EQUATION: \1', text)
        text = re.sub(r'\\\\([^\\]+)\\\\', r'EQUATION: \1', text)
        
        # Preserve numbered lists
        text = re.sub(r'(\d+)\.\s+', r'\1. ', text)
        
        # Preserve bullet lists
        text = re.sub(r'•\s+', '• ', text)
        
        return text
    
    def normalize_content(self, document):
        """
        Normalize document content regardless of source format.
        
        Args:
            document (dict): Document to normalize
            
        Returns:
            dict: Normalized document
        """
        # Deep copy to avoid modifying original
        normalized = document.copy()
        
        # Clean the raw text
        if 'raw_text' in normalized:
            normalized['raw_text'] = self.clean_text(normalized['raw_text'])
            normalized['raw_text'] = self.preserve_special_formatting(normalized['raw_text'])
        
        # Add additional metadata for content structure
        normalized['sections'] = self.extract_sections(normalized['raw_text'])
        
        return normalized
    
    def extract_sections(self, text):
        """
        Extract sections based on heading patterns.
        
        Args:
            text (str): Document text
            
        Returns:
            list: List of sections with headings and content
        """
        # Simple section extraction based on line starts with number or capital letters
        # This is a basic implementation and might need refinement
        section_pattern = re.compile(r'^((?:\d+\.)+\s+[A-Z][^.]+|[A-Z][^.]+:)', re.MULTILINE)
        
        # Find all section starts
        section_matches = list(section_pattern.finditer(text))
        sections = []
        
        if not section_matches:
            # If no sections found, return the whole text as one section
            return [{"heading": "", "content": text}]
        
        # Extract each section
        for i, match in enumerate(section_matches):
            start_pos = match.start()
            heading = match.group(0).strip()
            
            # Determine section end
            if i < len(section_matches) - 1:
                end_pos = section_matches[i + 1].start()
                content = text[start_pos + len(heading):end_pos].strip()
            else:
                content = text[start_pos + len(heading):].strip()
            
            sections.append({
                "heading": heading,
                "content": content
            })
        
        return sections
    
    def process_all_documents(self):
        """
        Process all documents in the input directory.
        """
        # Get all JSON files in the input directory
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        
        with tqdm(total=len(json_files), desc="Processing documents") as pbar:
            for json_file in json_files:
                try:
                    # Read the document
                    with open(json_file, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    # Normalize the document
                    normalized = self.normalize_content(document)
                    
                    # Save the processed document
                    output_file = os.path.join(self.output_dir, os.path.basename(json_file))
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(normalized, f, ensure_ascii=False, indent=2)
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    pbar.update(1)
        
        print(f"Processing complete. Processed {len(json_files)} documents.")


if __name__ == "__main__":
    # Create and run the processor
    processor = ContentProcessor()
    processor.process_all_documents()