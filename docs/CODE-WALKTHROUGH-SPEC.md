# Code Walkthrough — Authoring Spec

> Saved as a standing template. Applied to the source files when they arrive.
> Target deliverable: `<project>-CODE-WALKTHROUGH.md`

## Output file structure

1. **Overview**
   - What the program does, in 3–5 sentences
   - Language, frameworks, and libraries used
   - How the program flows from start to finish

2. **Imports and Dependencies**
   - Table: `Import | What it is | Why it's used here`

3. **Function and Class Reference**
   For each function or class:
   - **Name and signature**
   - **Purpose** — what it does and why it exists
   - **Parameters** — name, type, meaning
   - **Returns** — type and meaning
   - **Side effects** — file I/O, network calls, global state changes, etc.
   - **Called by / Calls** — how it connects to the rest of the code

4. **Line-by-Line Walkthrough**
   - Code in order, chunked along natural boundaries (one function or block at a time)
   - Original code in a fenced block with the correct language tag and line numbers
   - Directly below, every line explained as:
     `Line N: <code> → <plain-English explanation of what it does and why>`
   - No skipping. Blank lines and closing braces may be grouped:
     "Lines 12–14: closing braces for the loop and function."

5. **Potential Issues and Improvements**
   - Bugs, edge cases, unclear logic, security concerns, performance problems
   - Reference line numbers for each point

6. **Glossary**
   - Technical terms and concepts, explained simply

## Rules

- Explain the *why*, not just the *what*. "Increments i by 1" is insufficient — say what it accomplishes in context.
- Assume a beginner reader, without being condescending.
- Use the exact variable, function, and class names from the source.
- Do not invent behavior. Flag anything unclear or dependent on unseen code explicitly.
- Do not modify, refactor, or rewrite the code outside the "Potential Issues and Improvements" section.
- If the code is too long for one pass, stop at a clean boundary (end of a function or class), state where it stopped, and continue on "continue".
- If multiple files: add a short section on how the files relate, then document each file separately under its own heading.
- Output only the Markdown content, no extra commentary before or after.
