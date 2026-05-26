from bs4 import BeautifulSoup
import os


def parse_xhtml(file_path: str) -> list[dict]:

    source = os.path.basename(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), features="xml")
    
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    paragraphs = soup.get_text(separator="\n").split("\n")
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= 20]
    
    pages = []
    page = ""
    i = 0

    for paragraph in paragraphs:

        page += paragraph
        i += 1

        if i//10 :

            pages.append(page)
            page = ""
    
    pages = []

    for i in range(0, len(paragraphs), 5):

        group = " ".join(paragraphs[i:i+5])
        pages.append({
            "source": source,
            "page_number": i // 5 + 1,
            "text": group
        })

    return pages


if __name__== "__main__":

    pages = parse_xhtml(r"D:\Projects\amf-rag-agent\data\raw\529900S21EQ1BO4ESM68-2025-12-31-1-fr.xhtml")
    print(len(pages))
    print(pages[500])