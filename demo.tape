# vstash demo — terminal GIF for README
Output demo.gif

Set Shell "zsh"
Set FontSize 16
Set Width 900
Set Height 500
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 40ms

# Ingest a real URL — shows the full pipeline
Type "vstash add https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
Enter
Sleep 15s

# List to confirm
Type "vstash list"
Enter
Sleep 2s

# Ask a meaningful question
Type 'vstash ask "How does RAG reduce hallucinations?" --no-stream'
Enter
Sleep 12s

# Show stats
Type "vstash stats"
Enter
Sleep 3s
