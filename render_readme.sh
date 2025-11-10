#!/bin/bash
# Render README_ALL.md to HTML using various methods
# Choose the method that works best for your system

echo "=== Hemibrain Connectomes Analysis - README Renderer ==="
echo ""

# Check if README_ALL.md exists
if [ ! -f "README_ALL.md" ]; then
    echo "❌ Error: README_ALL.md not found!"
    exit 1
fi

echo "📄 Found README_ALL.md ($(wc -l < README_ALL.md) lines)"
echo ""
echo "Choose rendering method:"
echo "  1. Grip (GitHub-flavored, requires: pip install grip)"
echo "  2. Pandoc (powerful, requires: brew install pandoc)"
echo "  3. Markdown (simple, requires: pip install markdown)"
echo "  4. Python-Markdown with GitHub CSS (best formatting)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Using Grip..."
        if ! command -v grip &> /dev/null; then
            echo "Installing grip..."
            pip install grip
        fi
        grip README_ALL.md --export README_ALL.html
        echo "✅ Created: README_ALL.html"
        echo "Opening in browser..."
        open README_ALL.html 2>/dev/null || xdg-open README_ALL.html 2>/dev/null || start README_ALL.html
        ;;
    
    2)
        echo ""
        echo "Using Pandoc..."
        if ! command -v pandoc &> /dev/null; then
            echo "Installing pandoc..."
            brew install pandoc 2>/dev/null || sudo apt-get install pandoc
        fi
        pandoc README_ALL.md -o README_ALL.html --standalone --css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css --metadata title="Hemibrain Connectomes Analysis - Complete Documentation"
        echo "✅ Created: README_ALL.html"
        echo "Opening in browser..."
        open README_ALL.html 2>/dev/null || xdg-open README_ALL.html 2>/dev/null || start README_ALL.html
        ;;
    
    3)
        echo ""
        echo "Using Python-Markdown (simple)..."
        python3 << 'EOF'
import markdown
import sys

try:
    with open('README_ALL.md', 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    html = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite'])
    
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hemibrain Connectomes Analysis - Complete Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #24292e;
        }}
        h1, h2, h3, h4, h5, h6 {{
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
        }}
        pre {{
            background: #f6f8fa;
            padding: 16px;
            overflow: auto;
            border-radius: 6px;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
        }}
        blockquote {{
            border-left: 4px solid #dfe2e5;
            padding-left: 16px;
            color: #6a737d;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
    
    with open('README_ALL.html', 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print("✅ Created: README_ALL.html")
    
except ImportError:
    print("❌ Error: markdown package not found")
    print("Installing markdown...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "markdown"])
    print("Please run this script again.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF
        echo "Opening in browser..."
        open README_ALL.html 2>/dev/null || xdg-open README_ALL.html 2>/dev/null || start README_ALL.html
        ;;
    
    4)
        echo ""
        echo "Using Python-Markdown with GitHub CSS (best formatting)..."
        python3 << 'EOF'
import sys

try:
    import markdown
except ImportError:
    print("Installing markdown...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

with open('README_ALL.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

# Convert markdown to HTML with extensions
html_content = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc', 'attr_list']
)

# Full HTML with GitHub styling
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hemibrain Connectomes Analysis - Complete Documentation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css">
    <style>
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
        }}
        @media (max-width: 767px) {{
            .markdown-body {{
                padding: 15px;
            }}
        }}
        /* Custom enhancements */
        .markdown-body pre {{
            background-color: #f6f8fa;
        }}
        .markdown-body code {{
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            padding: 0.2em 0.4em;
        }}
        /* Table of contents */
        #toc {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 24px;
        }}
    </style>
</head>
<body>
    <article class="markdown-body">
        {html_content}
    </article>
</body>
</html>
"""

with open('README_ALL.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("✅ Created: README_ALL.html with GitHub styling")
print("📊 File size:", len(full_html), "bytes")
EOF
        echo "Opening in browser..."
        open README_ALL.html 2>/dev/null || xdg-open README_ALL.html 2>/dev/null || start README_ALL.html
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✨ Done! You can now:"
echo "   - View README_ALL.html in any browser"
echo "   - Print to PDF from the browser (Cmd/Ctrl+P)"
echo "   - Share the HTML file"
echo ""
