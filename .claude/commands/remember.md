---
argument-hint: <file, directory, or URL>
description: Add documents to vstash memory for future context
---

# vstash Remember

Ingest files, directories, or URLs into vstash so they're available for future `/memory` searches.

## Instructions

1. Run `vstash add` with the provided source using the Bash tool:

```bash
vstash add $ARGUMENTS
```

2. If the user provides multiple sources, pass them all at once:
```bash
vstash add file1.pdf file2.md https://example.com/article
```

3. If the user specifies a collection or project, pass the flags:
```bash
vstash add $ARGUMENTS --collection <name> --project <name>
```

4. After ingestion, confirm what was added with a brief summary (title, chunks, collection).

5. If the user asks to remember a directory, warn them about large directories and suggest using `--collection` to organize.
