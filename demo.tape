# vstash demo — terminal GIF for README
Output demo.gif

Set Shell "zsh"
Set FontSize 15
Set Width 1000
Set Height 550
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 35ms

# --- 1. Add a Python file ---
Type "vstash add vstash/ingest.py"
Enter
Sleep 8s

# --- 2. Search without LLM (free, instant) ---
Type 'vstash search "SSRF validation"'
Enter
Sleep 5s

# --- 3. Add a URL ---
Type "vstash add https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
Enter
Sleep 15s

# --- 4. Ask a question (LLM answer with sources) ---
Type 'vstash ask "How does RAG reduce hallucinations?" --no-stream'
Enter
Sleep 12s

# --- 5. Stats ---
Type "vstash stats"
Enter
Sleep 3s
