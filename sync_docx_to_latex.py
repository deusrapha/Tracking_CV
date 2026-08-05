import os
import re
import docx
import requests
import pypdf

# Helper function to escape LaTeX special characters
def escape_latex(text):
    # Map of special characters that need to be escaped in LaTeX
    # Note: we should not escape LaTeX commands if the user writes them, 
    # but for standard text editing, we should escape common characters.
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    
    # First, let's check if there are any math equations in the paragraph.
    # If the user put equations like $ExG = 2G - R - B$, we want to preserve the $ characters.
    # We can split the text by $ and escape odd parts (which are standard text) and keep even parts (which are math).
    parts = text.split('$')
    for i in range(len(parts)):
        if i % 2 == 0: # Standard text part, escape characters
            for char, escaped in special_chars.items():
                parts[i] = parts[i].replace(char, escaped)
        else: # Math part, only escape percent sign if any
            parts[i] = parts[i].replace('%', r'\%')
            
    return '$'.join(parts)

def parse_docx_proposal(docx_path):
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Word document not found: {docx_path}")
        
    doc = docx.Document(docx_path)
    sections = {}
    current_section = None
    section_paragraphs = []
    
    # 1. Parse paragraphs for standard text sections
    for p in doc.paragraphs:
        text = p.text.strip()
        if text.startswith("[[") and text.endswith("]]"):
            # Save previous section
            if current_section:
                sections[current_section] = "\n\n".join(section_paragraphs)
                
            # Start new section
            current_section = text[2:-2].strip()
            section_paragraphs = []
            print(f"Detected section in Word document: {current_section}")
        else:
            if current_section and text:
                section_paragraphs.append(escape_latex(text))
                
    # Save the last section
    if current_section:
        sections[current_section] = "\n\n".join(section_paragraphs)

    # 2. Parse tables specifically for Appendix A and B
    # Appendix A: Itemized Budget
    # Let's look for tables in the document and identify the budget table
    for table in doc.tables:
        if len(table.columns) == 3: # Budget table has 3 columns (Category, Item Description, Cost)
            # Verify if it's the budget table by checking headers
            hdr_text = [cell.text.lower() for cell in table.rows[0].cells]
            if 'category' in hdr_text[0] or 'cost' in hdr_text[2]:
                print("Detected Appendix A: Budget table in Word document.")
                budget_rows = []
                for row in table.rows[1:]: # Skip header
                    cells = [c.text.strip() for c in row.cells]
                    # Escape LaTeX special chars
                    cat = escape_latex(cells[0])
                    desc = escape_latex(cells[1])
                    cost = escape_latex(cells[2])
                    
                    if cat == 'Total':
                        budget_rows.append(r"    \textbf{Total} & & \textbf{" + cost + r"} \\")
                    else:
                        budget_rows.append(f"    \\textbf{{{cat}}} & {desc} & {cost} \\\\")
                sections["TABLE: Budget"] = "\n".join(budget_rows)
                
        elif len(table.columns) == 5: # Gantt Chart table has 5 columns
            hdr_text = [cell.text.lower() for cell in table.rows[0].cells]
            if 'activity' in hdr_text[0] or 'q1' in hdr_text[1]:
                print("Detected Appendix B: Work Plan table in Word document.")
                wp_rows = []
                for row in table.rows[1:]: # Skip header
                    cells = [escape_latex(c.text.strip()) for c in row.cells]
                    wp_rows.append("    " + " & ".join(cells) + " \\\\")
                sections["TABLE: WorkPlan"] = "\n".join(wp_rows)

    return sections

def update_latex_file(tex_path, sections):
    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"LaTeX file not found: {tex_path}")
        
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    for sec_name, sec_content in sections.items():
        start_tag = f"% [[START: {sec_name}]]"
        end_tag = f"% [[END: {sec_name}]]"
        
        # We search for the start and end tags in LaTeX file
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, content, re.DOTALL)
        if match:
            print(f"Updating LaTeX section: {sec_name}")
            # Replace matching content
            content = re.sub(pattern, f"{start_tag}\n{sec_content}\n{end_tag}", content, flags=re.DOTALL)
        else:
            print(f"WARNING: Tag {start_tag} / {end_tag} not found in LaTeX file.")
            
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"LaTeX file successfully updated: {tex_path}")

def compile_latex(tex_path, pdf_path):
    print("Compiling updated LaTeX file to PDF via online API...")
    with open(tex_path, "r", encoding="utf-8") as f:
        latex_text = f.read()
        
    url = "https://latex.ytotech.com/builds/sync"
    payload = {
        "compiler": "pdflatex",
        "resources": [
            {
                "main": True,
                "content": latex_text
            }
        ]
    }
    
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code == 201:
        print(f"Compilation successful! Saving PDF to: {pdf_path}")
        with open(pdf_path, "wb") as f:
            f.write(response.content)
            
        reader = pypdf.PdfReader(pdf_path)
        print(f"Successfully compiled. New page count: {len(reader.pages)}")
    else:
        print(f"Compilation failed with status code: {response.status_code}")
        print(response.text[:1000])

def main():
    print("--------------------------------------------------")
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docx_path = os.path.join(script_dir, "Research_Proposal.docx")
    tex_path = os.path.join(script_dir, "Research_Proposal.tex")
    pdf_path = os.path.join(script_dir, "Research_Proposal.pdf")
    
    print("Parsing edited Word document...")
    try:
        sections = parse_docx_proposal(docx_path)
        print("\nUpdating LaTeX source...")
        update_latex_file(tex_path, sections)
        print("\nCompiling updated document...")
        compile_latex(tex_path, pdf_path)
        print("\nSync completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
