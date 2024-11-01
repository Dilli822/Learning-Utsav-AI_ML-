from bs4 import BeautifulSoup

def extract_all_information_from_html(file_path):
    """
    Extracts all textual information from a local HTML file, 
    processing all tags to produce structured output while excluding
    <style> and <script> tags.
    
    Args:
        file_path (str): The path to the HTML file.
    
    Returns:
        dict: A dictionary containing the extracted formatted text.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    
    # Attempt to extract text from the element with ID 'mw-content-text'
    body_content = soup.find(id="mw-content-text")  # Change this based on your HTML structure
    if body_content:
        # Remove <style> and <script> tags
        for script in body_content(['script', 'style']):
            script.decompose()  # Remove these elements from the soup

        formatted_text = ""
        
        # Iterate over all elements in the body content
        for element in body_content.find_all(True):  # True means all tags
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                formatted_text += f"{element.get_text()}\n{'=' * len(element.get_text())}\n\n"
            elif element.name == 'p':
                formatted_text += f"{element.get_text()}\n\n"
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    formatted_text += f"- {li.get_text()}\n"
                formatted_text += "\n"  # Add a newline after each list
            elif element.name == 'ol':
                for index, li in enumerate(element.find_all('li'), start=1):
                    formatted_text += f"{index}. {li.get_text()}\n"
                formatted_text += "\n"  # Add a newline after each list
            elif element.name == 'blockquote':
                formatted_text += f"> {element.get_text()}\n\n"
            elif element.name == 'dl':
                for dt in element.find_all('dt'):
                    formatted_text += f"{dt.get_text()}:\n"
                    for dd in dt.find_next_siblings('dd'):
                        formatted_text += f"  - {dd.get_text()}\n"
                formatted_text += "\n"  # Add a newline after each definition list
            else:
                # For other tags, simply get the text and add it to the formatted text
                if element.get_text(strip=True):  # Avoid adding empty text
                    formatted_text += f"{element.get_text(strip=True)}\n"

        content_text = formatted_text.strip()
    else:
        content_text = "No content found."

    return {
        "Content": content_text
    }

def save_to_file(content_info, filename):
    """
    Saves the extracted information to a text file.
    
    Args:
        content_info (dict): The dictionary containing the extracted data.
        filename (str): The name of the file to save the data.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Content:\n{content_info['Content']}\n")

# Example usage
file_path = "diabetes.html"  # Replace with your HTML file path
extracted_info = extract_all_information_from_html(file_path)

if extracted_info:
    save_to_file(extracted_info, "extracted_content.txt")
    print("Data saved to extracted_content.txt")
