import asyncio
import markdown
from playwright.async_api import async_playwright

async def generate_pdf_bytes(messages_data):
    # 1. Build rich HTML with MathJax for LaTeX and CSS for tables
    html_content = """
    
    
        
        
        
        
        
    
    
        🏗️ CivilGPT Engineering Notes
    """
    
    for msg in messages_data:
        if msg["role"] == "user":
            html_content += f"🧑‍🎓 Question: {msg['content']}"
        else:
            # Convert markdown (including tables) to HTML
            parsed_md = markdown.markdown(msg["content"], extensions=['tables', 'fenced_code'])
            html_content += f"{parsed_md}"
            
            # Append sources if available
            if msg.get("sources"):
                html_content += "📚 Sources:"
                for src in msg["sources"]:
                    html_content += f"{src['file']} (Page {src['page']})"
                html_content += ""

    html_content += ""

    # 2. Use Playwright to render the HTML into a PDF in memory
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Load the HTML
        await page.set_content(html_content)
        
        # Wait for MathJax to finish drawing the formulas before taking the PDF snapshot
        await page.wait_for_timeout(2000) 
        
        # Generate PDF bytes
        pdf_bytes = await page.pdf(format="A4", print_background=True, margin={"top": "20px", "bottom": "20px"})
        await browser.close()
        
        return pdf_bytes