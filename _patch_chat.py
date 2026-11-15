import pathlib

p = pathlib.Path("bern/engine.py")
s = p.read_text()

# ---- 1. imports ------------------------------------------------------------
old_imp = "from bern.trainer import load_dataset, save_dataset, train_bern_model"
new_imp = ("from bern.conversation import Memory, wants_explanation\n"
           "from bern.trainer import load_dataset, save_dataset, train_bern_model")
assert old_imp in s, "trainer import not found"
s = s.replace(old_imp, new_imp, 1)

# ---- 2. session initialisation carries a Memory ----------------------------
old_session = '''        session = self.sessions.setdefault(session_id, {"user_name": None, "turns": 0})
        session["turns"] += 1'''
new_session = '''        session = self.sessions.setdefault(
            session_id, {"user_name": None, "turns": 0, "memory": Memory()}
        )
        session["turns"] += 1'''
assert old_session in s, "session init not found"
s = s.replace(old_session, new_session, 1)

# ---- 3. rename the existing body to _answer --------------------------------
old_def = '''    def chat(self, message: str, session_id: str = "default") -> dict[str, Any]:
        """Generate Bern's response plus full ML prediction metadata."""'''
new_def = '''    def chat(self, message: str, session_id: str = "default", engine: str = "auto",
             use_llm: bool = True) -> dict[str, Any]:
        """Answer one message, using what was said earlier in this session.

        Three things happen here that a plain classifier cannot do:

        * **Memory.** Session transcripts are kept, and a follow-up such as "what about the
          other one?" is rewritten into the question it stands for before classification.
        * **Unseen questions.** When local routing declines a message and a language model is
          configured, the model answers instead of Bern giving up.
        * **Explanations.** "Bakit mahal ang bigas?" asks for a reason; a bag-of-words model
          has none to give, so a configured model answers it.

        Args:
            message: What the user typed.
            session_id: Conversation to continue. Sessions are in-memory and per-process.
            engine: ``"local"`` never uses a language model; ``"llm"`` uses it for the cases
                above and falls back locally if the call fails; ``"auto"`` is ``"llm"`` with
                the same fallback. No key means the local path either way.
            use_llm: Set False for deterministic, offline runs — the frozen benchmark uses
                it, so scoring never depends on a network call.

        Returns:
            The usual reply payload, plus ``conversation`` (what is remembered) and
            ``resolved_as`` (the rewritten question, when one was rewritten).
        """
        memory: Memory = self.sessions.setdefault(
            session_id, {"user_name": None, "turns": 0, "memory": Memory()}
        ).setdefault("memory", Memory())

        resolved, subject = memory.resolve(message)
        result = self._answer(resolved, session_id)

        wants_reason = wants_explanation(message)
        if use_llm and engine in ("llm", "auto") and (result.get("needs_training") or wants_reason):
            generated = self._llm_answer(message, result, subject)
            if generated is not None:
                result = generated

        # Only record what actually happened: a declined question must not become the
        # subject of the next "what about that one?".
        if not result.get("needs_training"):
            memory.record(message, str(result.get("intent", "")), str(result.get("reply", "")),
                          list(result.get("entities", [])))

        result["conversation"] = memory.summary()
        result["resolved_as"] = resolved if subject else None
        return result

    def _llm_answer(self, message: str, local: dict[str, Any], subject: str | None) -> dict[str, Any] | None:
        """Try the language model; return None to keep the local answer.

        The local answer is passed in as grounding so the model can quote Bern's real price
        and nutrition data instead of recalling it from training and getting it wrong.
        """
        try:
            from bern import llm
        except Exception:
            return None
        if not llm.is_available():
            return None

        context = ""
        if local.get("intent") not in (None, "unknown") and not wants_explanation(message):
            context = str(local.get("reply", ""))[:1500]

        question = message if subject is None else f"{message} (about {subject})"
        try:
            session = llm.TutorSession(notes=context)
            answer = session.ask(question)["answer"].strip()
        except Exception:
            # A model that is down must not take the chat down with it.
            return None
        if not answer:
            return None

        return {
            "reply": answer,
            "intent": local.get("intent", "unknown"),
            "confidence": local.get("confidence", 0.0),
            "needs_training": False,
            "engine": "llm",
            "top_predictions": local.get("top_predictions", []),
            "top_signals": local.get("top_signals", []),
            "inference_ms": local.get("inference_ms", 0.0),
            "ai_name": self.bundle.get("ai_name", "Bern"),
            "model_version": self.bundle.get("version", "1.0.0"),
            "trained_at": self.bundle.get("trained_at"),
        }

    def _answer(self, message: str, session_id: str = "default") -> dict[str, Any]:
        """Single-message routing. No memory, no language model. See :meth:`chat`."""'''
assert old_def in s, "chat def not found"
s = s.replace(old_def, new_def, 1)

p.write_text(s)
print("engine.chat wrapped")
