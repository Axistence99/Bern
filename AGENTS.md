# AGENTS.md — Standing instructions for this repository

These are the project owner's instructions, reproduced **verbatim**. They are permanent
unless the owner revises them. Read them before changing anything in this repository.

---

## 0. Universal Vibe Coding Rules

Before changing anything, inspect the existing project structure, relevant files, dependencies, and current behavior.

Do not blindly rewrite the project.

Follow these rules:

- Do not remove existing functionality unless explicitly requested.
- Reuse existing components, utilities, styles, and architecture when practical.
- Make the smallest reasonable change needed.
- Preserve the current architecture unless there is a clear reason to improve it.
- Do not add dependencies unless they are genuinely necessary.
- Do not duplicate existing functionality.
- Keep code readable and maintainable.
- Avoid unnecessary abstraction and overengineering.
- Preserve existing UI/UX unless the requested change requires modifying it.
- Consider security, accessibility, responsiveness, performance, and error handling.

After implementing changes:

- Check syntax and imports.
- Check for runtime errors.
- Check browser console errors where applicable.
- Verify the requested feature.
- Verify that existing functionality still works.
- Check responsive behavior.
- Check accessibility.
- Check security implications.
- Check performance implications.

Never claim that something was tested if you did not actually test it.

Prioritize:

1. Correctness
2. Security
3. Maintainability
4. Accessibility
5. Performance
6. User experience
7. Visual polish

---

## 1. Project Documentation

Inspect the entire project and create or update comprehensive documentation.

Document the actual implementation, not an assumed implementation.

Include:

- Project overview
- Main features
- Technology stack
- Project structure
- Important files and their responsibilities
- Installation/setup instructions
- Environment variables and configuration
- How to run the project
- How to build the project
- How to test the project
- API endpoints if applicable
- Database structure if applicable
- Deployment instructions
- Important development notes
- Known limitations

Keep the documentation synchronized with the current codebase.

Do not invent features, commands, files, APIs, or configuration that do not exist.

---

## 2. Human-Understandable Code

Review the code and make it easier for a human developer to understand.

Improve:

- Variable names
- Function/method names
- Class names
- File organization
- Function structure
- Logical grouping
- Complex conditions
- Repeated logic

Add comments only where they provide useful context, especially for:

- Complex algorithms
- Non-obvious business logic
- Security-sensitive logic
- Important workarounds
- External API behavior

Do not add meaningless comments that simply restate the code.

Keep the behavior unchanged unless a bug is discovered and fixing it is necessary.

---

## 3. Line-by-Line Code Walkthrough

Full authoring template: [`docs/CODE-WALKTHROUGH-SPEC.md`](docs/CODE-WALKTHROUGH-SPEC.md).
Summary of the binding rules:

- Explain the *why*, not just the *what*.
- Assume a beginner reader, without being condescending.
- Use the exact variable, function, and class names from the source.
- Do not invent behavior. Flag anything unclear or dependent on unseen code explicitly.
- Do not modify, refactor, or rewrite the code outside the
  "Potential Issues and Improvements" section.
- Output only the Markdown content, no extra commentary before or after.

---

## 4. Absolute environment constraint

**Never run a sandbox server.** No long-lived processes, no dev servers, no watchers, no
`npm run dev`, no `python3 server.py`. One-shot commands only.

Consequence: HTTP handlers may be statically checked and unit-tested but cannot be exercised
over a socket. Never report an HTTP route as working unless the owner lifts this rule and it
is actually exercised.

---

## 5. Known conflict between §2 and §3

§2 permits (and requests) refactoring for readability. §3 forbids modifying or refactoring
code outside the "Potential Issues and Improvements" section. These cannot both be satisfied
in one pass.

**Default when the owner does not say otherwise:** treat §3 as active — leave code untouched,
put refactor suggestions in the issues section. If the owner asks for both, do the refactor
first, then document the *final* code so line numbers stay accurate.

---

## 6. Contribution convention (owner-set)

- **Commit author:** `Axistence99 <272527416+Axistence99@users.noreply.github.com>`
  (GitHub noreply address — links commits to the owner's profile without exposing a
  private email). Never commit as an agent or bot identity.
- **Branch:** always commit and push to **`main`**. Do not open feature branches, do not
  leave work on a side branch, do not ask which branch to use.
- Set the identity in a fresh sandbox before committing, because `.git/config` is not
  carried over:
  ```bash
  git config user.name  "Axistence99"
  git config user.email "272527416+Axistence99@users.noreply.github.com"
  ```
