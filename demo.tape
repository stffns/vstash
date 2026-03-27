# vstash demo — terminal GIF for README
Output demo.gif

Set Shell "zsh"
Set FontSize 15
Set Width 1000
Set Height 550
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 35ms

# --- 1. Add a Python file (shows code-aware chunking) ---
Type "# Add a Python file — chunks by function boundaries"
Enter
Sleep 800ms
Type "vstash add vstash/ingest.py"
Enter
Sleep 8s

# --- 2. Search without LLM (free, instant) ---
Type "# Search is 100% local — no API key, no LLM"
Enter
Sleep 800ms
Type 'vstash search "SSRF validation"'
Enter
Sleep 5s

# --- 3. Add a URL (shows versatility) ---
Type "# Add any file, directory, or URL"
Enter
Sleep 800ms
Type "vstash add https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
Enter
Sleep 15s

# --- 4. Ask a question (shows LLM answer with sources) ---
Type "# Ask gets an LLM-powered answer with cited sources"
Enter
Sleep 800ms
Type 'vstash ask "How does RAG reduce hallucinations?" --no-stream'
Enter
Sleep 12s

# --- 5. Stats ---
Type "vstash stats"
Enter
Sleep 3s
