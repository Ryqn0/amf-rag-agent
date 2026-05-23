import pymupdf


def parse_pdf(file_path: str) -> list[dict] :
    """Parse a PDF document and return its content as a list of dictionaries.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        list[dict]: A list of dictionaries containing the parsed content.
    """
    
    doc = pymupdf.open(file_path)

    n = doc.page_count

    parsed = []
    for i in range(n):
        parsed.append(
            {
                "source": file_path,
                "page_number": i + 1,
                "text" : doc[i].get_text(),
            }
        )

    return parsed

if __name__ == "__main__":
    pages = parse_pdf("data/raw/totalenergies_universal-registration-document-2025_2026_en.pdf")
    print(f"Parsed {len(pages)}")
    print(f"Page 6 preview: {pages[5]['text'][:300]}")