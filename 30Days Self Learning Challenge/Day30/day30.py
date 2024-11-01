import requests
from bs4 import BeautifulSoup

def scrape_medical_data(disease_name):
    """
    Scrapes formatted textual information from a specific element on a given disease page on Wikipedia.
    
    Args:
        disease_name (str): The name of the disease to search for.
    
    Returns:
        dict: A dictionary containing the disease name and the extracted formatted text.
    """
    base_url = "https://en.wikipedia.org/wiki/"
    url = base_url + disease_name.replace(" ", "_")
    
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve data")
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    
    # Attempt to extract text from the element with ID 'mw-content-text'
    body_content = soup.find(id="mw-content-text")
    if body_content:
        # Extract paragraphs, headings, and lists
        formatted_text = ""
        
        # Process headings
        for heading in body_content.find_all(['h1', 'h2', 'h3']):
            formatted_text += f"{heading.get_text()}\n{'=' * len(heading.get_text())}\n\n"
        
        # Process paragraphs
        for paragraph in body_content.find_all('p'):
            formatted_text += f"{paragraph.get_text()}\n\n"
        
        # Process unordered lists
        for ul in body_content.find_all('ul'):
            for li in ul.find_all('li'):
                formatted_text += f"- {li.get_text()}\n"
            formatted_text += "\n"  # Add a newline after each list
        
        # Process ordered lists
        for ol in body_content.find_all('ol'):
            for li in ol.find_all('li'):
                formatted_text += f"1. {li.get_text()}\n"  # You can modify numbering as needed
            formatted_text += "\n"  # Add a newline after each list
        
        content_text = formatted_text.strip()
    else:
        content_text = "No content found."

    return {
        "Disease": disease_name,
        "Content": content_text
    }

def save_to_file(disease_info, filename):
    """
    Saves the disease information to a text file.
    
    Args:
        disease_info (dict): The dictionary containing disease data.
        filename (str): The name of the file to save the data.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Disease: {disease_info['Disease']}\n\n")
        file.write(f"Content:\n{disease_info['Content']}\n")

# Example usage
disease_name = "Diabetes"
disease_info = scrape_medical_data(disease_name)

if disease_info:
    save_to_file(disease_info, f"{disease_name}.txt")
    print(f"Data saved to {disease_name}.txt")
