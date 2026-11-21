# De-slop pass — audit and direction

## STEP 1: AUDIT

Measured from the current `web/demo/app.jsx`, not from memory:

| # | Tell | Evidence in the build |
| --- | --- | --- |
| 1 | **Purple-to-violet brand gradient** | `bg-gradient-to-br from-indigo-500 to-violet-600` on the logo mark, plus `shadow-indigo-500/25`, `shadow-indigo-600/20` glow behind it. The single most templated choice in the file. |
| 2 | **Five corner radii, three shadow depths** | `rounded-md`, `-lg`, `-xl`, `-2xl`, `-full` — 6 / 5 / 2 / 3 / 8 uses respectively, with `shadow-sm`, `shadow-md`, `shadow-lg`. Nothing decides; everything is softened. |
| 3 | **Six semantic colors competing** | indigo (brand + send), emerald (ready + confidence), rose (out-of-scope + low confidence + error), amber (server note + loading), plus slate tones. Four different hues all signalling "state". |
| 4 | **Default system font, no hierarchy** | `-apple-system, BlinkMacSystemFont, "Segoe UI"` — the same stack as every dashboard template. Every size between 11 px and 18 px is in use; nothing is *large*. The h1 is 16 px, the same as body copy. |
| 5 | **Everything is a rounded card with a soft shadow** | The prompt card, each reply bubble, the score panel, the note strip, the buttons — all the same construction. Importance is expressed only by tint. |
| 6 | **Pills as decoration** | Three `rounded-full` pills in the header (status, and the same shape reused for the score and the intent tag). A pill is a promise that the content is a discrete, countable status — it is not. |
| 7 | **Vague, self-describing copy** | "Try it yourself", "Intent model, running in this tab", "Type anything." None of it names the thing on screen. No numbers a visitor could check. |
| 8 | **Even, safe spacing** | One padding value, one rhythm, no focal point. The eye enters at the top-left card and has nowhere in particular to land. |

Verdict: with the mascot removed, this screen could be sold as a generic "AI demo" template.
That is the test it fails.

---

## STEP 2: DIRECTION

**A laboratory bench, not a chat toy.** This is for the person who wants to see whether the
model is actually any good — a developer, a classmate, a client being shown what was built. It
should feel like a measurement instrument: unsentimental, legible, and honest about uncertainty.

The single idea: **the confidence number is the interface.** Every reply is stamped with what
the model decided and how sure it was, in a fixed, tabular column — so the thing you read first
is how much the machine believes itself, not how friendly it sounds.

Consequences that follow from that, and that later choices must serve:

- Numbers get the typographic weight, not the prose. A 40 px serif percentage is the focal point.
- Colour is a signal, never decoration: **one** accent (a desaturated signal green) for "it knows",
  **one** warning (burnt orange) for "it is guessing" and "this needs the server". Nothing else.
- Rules and spacing carry structure. No shadows. No gradients. Radius 0 or 2 px.
- The copy states what the model did in measurable terms, not what it is.
