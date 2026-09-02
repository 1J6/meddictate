#!/usr/bin/env python3
"""
corrector.py - optional spelling-repair stage for meddictate, fully offline.

MedASR spells unfamiliar words phonetically ("sofwhere", "orrsodoxycholic").
This stage asks a local LLM (MedGemma via llama.cpp, running on this machine
only) to repair them, then applies a DICTIONARY GUARD: the model's edits are
accepted only where the original words are NOT in the English + medical word
lists. Numbers, units, doses, abbreviations and every known word are pasted
exactly as MedASR produced them, so the LLM can never silently change a dose
or swap a drug that was heard correctly.

If every word in a phrase is already known, the LLM is not called at all.

Files expected next to this script:
    llama\\llama-server.exe            llama.cpp Windows build (Vulkan)
    models\\medgemma-4b-it-Q4_K_M.gguf the language model
    wordlists\\english_words.txt       dwyl/english-words (Unlicense)
    wordlists\\medical_terms.txt       glutanimate/wordlist-medicalterms-en (GPLv3)
    wordlists\\custom_words.txt        your own additions, one per line
"""

import difflib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LLAMA_SERVER = HERE / "llama" / "llama-server.exe"
MODEL = HERE / "models" / "medgemma-4b-it-Q4_K_M.gguf"
WORDLISTS = [
    HERE / "wordlists" / "english_words.txt",
    HERE / "wordlists" / "medical_terms.txt",
    HERE / "wordlists" / "custom_words.txt",
]
PORT = 8765
GPU_LAYERS = 99            # offload everything to the GPU; set 0 to run on CPU only
CONTEXT = 2048
MAX_WORDS_PER_FIX = 3      # a non-word may become at most this many words (run-together speech)
MIN_SIMILARITY = 0.45      # ...and any replacement must still resemble the original word(s)
FALLBACK_MIN_SIMILARITY = 0.8    # dictionary-only fixes need this similarity to apply
REQUEST_TIMEOUT = 12       # seconds; on timeout the phrase is pasted uncorrected
STARTUP_TIMEOUT = 120      # seconds to wait for the model to load

SYSTEM_PROMPT = (
    "You correct spelling in medical dictation transcripts produced by speech "
    "recognition. Fix only words that are misspelled or are not real words, "
    "replacing each with the medical or everyday word that sounds most like it. "
    "A non-word may be several words run together; split it if so. "
    "Never change numbers, units, doses, punctuation, capitalization or word order. "
    "Never add, remove or reorder any other words. Never explain. "
    "Reply with the corrected transcript only."
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d[\w.,:/%-]*|\s+|[^\sA-Za-z\d]")


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _join(tokens) -> str:
    """Join tokens with spaces, but none before punctuation."""
    s = ""
    for t in tokens:
        if s and (t[0].isalnum()):
            s += " "
        s += t
    return s


def skeleton(word: str) -> str:
    """Crude sound-alike key: drop vowels and collapse repeats, so that
    'orrsodoxycholic' and 'ursodeoxycholic' both become 'rsdxchlc'."""
    s = re.sub(r"[^a-z]", "", word.lower())
    s = s.replace("ph", "f").replace("ck", "k").replace("qu", "k")
    s = re.sub(r"[aeiouyhw]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


class Corrector:
    def __init__(self, log=print):
        self.log = log
        self.words = set()
        self.proc = None
        self.ready = False
        self.failed = False
        self._job = None
        self.load_wordlists()

    # ---- dictionary --------------------------------------------------------

    def load_wordlists(self):
        n = 0
        self.by_skeleton = {}
        for path in WORDLISTS:
            if not path.is_file():
                continue
            medical = path.name != "english_words.txt"
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if not w:
                        continue
                    self.words.add(w)
                    n += 1
                    if len(w) >= 5:
                        self.by_skeleton.setdefault(skeleton(w), []).append((medical, w))
        if n == 0:
            raise FileNotFoundError("no word lists found in " + str(HERE / "wordlists"))
        self.log(f"Corrector: {len(self.words):,} dictionary words loaded")

    def candidates(self, word: str, n=4):
        """Dictionary words that sound like `word`: same consonant skeleton
        first (catches orrsodoxycholic -> ursodeoxycholic), medical terms
        preferred, ranked by string similarity."""
        w = word.lower()
        pool = list(self.by_skeleton.get(skeleton(w), []))
        if len(pool) < n:
            near = [x for L in range(len(w) - 2, len(w) + 3) if L >= 5
                    for x in self._by_len().get(L, [])]
            close = difflib.get_close_matches(w, near, n=n * 3, cutoff=0.78)
            pool += [(False, c) for c in close if c not in {p[1] for p in pool}]
        scored = sorted(pool, key=lambda p: (-difflib.SequenceMatcher(None, w, p[1]).ratio(),
                                             not p[0]))
        out = []
        for _, cand in scored:
            if cand not in out:
                out.append(cand)
            if len(out) == n:
                break
        return out

    def _by_len(self):
        if not hasattr(self, "_len_index"):
            self._len_index = {}
            for w in self.words:
                self._len_index.setdefault(len(w), []).append(w)
        return self._len_index

    def known(self, tok: str) -> bool:
        """True if the token must be left alone by the LLM."""
        if not tok[0].isalpha():
            return True                     # numbers, punctuation, whitespace
        if tok.isupper() and len(tok) <= 6:
            return True                     # abbreviations: PE, GERD, COPD, CT
        if any(ch.isupper() for ch in tok[1:]):
            return True                     # CamelCase/brand tokens: UpToDate, MedASR, tag fragments
        w = tok.lower().replace("’", "'")
        if w.endswith("'s"):
            w = w[:-2]
        if w in self.words:
            return True
        parts = [p for p in re.split(r"[-']", w) if p]
        return len(parts) > 1 and all(p in self.words for p in parts)

    def unknown_words(self, text: str):
        seen, out = set(), []
        for tok in _TOKEN.findall(text):
            if not self.known(tok) and tok.lower() not in seen:
                seen.add(tok.lower())
                out.append(tok)
        return out

    # ---- llama-server lifecycle ----------------------------------------------

    def start(self):
        """Launch llama-server in the background (returns immediately)."""
        if self.proc or self.failed:
            return
        if not LLAMA_SERVER.is_file() or not MODEL.is_file():
            self.failed = True
            self.log(f"Corrector: disabled ({LLAMA_SERVER.name} or model missing)")
            return
        cmd = [
            str(LLAMA_SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
            "-ngl", str(GPU_LAYERS), "-c", str(CONTEXT), "--no-webui",
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        logfile = open(HERE / "llama" / "llama-server.log", "w", encoding="utf-8", errors="ignore")
        self.proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                     creationflags=flags, cwd=str(LLAMA_SERVER.parent))
        self._tie_to_our_lifetime()
        self.started_at = time.time()
        self.log("Corrector: starting MedGemma (llama-server)...")

    def _tie_to_our_lifetime(self):
        """Windows job object: if we die for any reason, llama-server dies too."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in
                            ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                             "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                            ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            job = k32.CreateJobObjectW(None, None)
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x2000     # KILL_ON_JOB_CLOSE
            k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            k32.AssignProcessToJobObject(job, wintypes.HANDLE(self.proc._handle))
            self._job = job                                    # keep handle alive
        except Exception:
            pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        self.ready = False

    def wait_ready(self, timeout=STARTUP_TIMEOUT) -> bool:
        if self.ready:
            return True
        if not self.proc:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.failed = True
                self.log("Corrector: llama-server exited; see llama\\llama-server.log")
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if json.loads(r.read().decode()).get("status") == "ok":
                        if not self.ready:
                            self.ready = True
                            self.log(f"Corrector: ready ({time.time() - self.started_at:.0f}s)")
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    # ---- correction --------------------------------------------------------

    def ask_llm(self, text: str, unknown) -> str:
        hints = []
        for w in unknown:
            c = self.candidates(w)
            hints.append(f"{w} (perhaps: {', '.join(c)})" if c else w)
        body = json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    "Words that may be misspelled, with dictionary look-alikes where "
                    "available: " + "; ".join(hints) +
                    "\n\nTranscript:\n" + text},
            ],
            "temperature": 0,
            "max_tokens": max(64, len(text) // 2 + 32),
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            out = json.loads(r.read().decode())["choices"][0]["message"]["content"]
        return out.strip()

    def dictionary_fix(self, tok: str):
        """Closest dictionary word for a single non-word, if it is a near-certain
        match; else None. Used when the LLM's edit is rejected or unavailable."""
        if self.known(tok):
            return None
        cand = self.candidates(tok, n=1)
        if cand and _ratio(tok.lower(), cand[0]) >= FALLBACK_MIN_SIMILARITY:
            return cand[0].capitalize() if tok[0].isupper() else cand[0]
        return None

    def guard(self, original: str, proposed: str):
        """Merge the LLM's proposal into the original, accepting an edit only
        where every original word is unknown, no words are invented, and no
        numbers or punctuation change. Original whitespace (spaces, newlines)
        is kept exactly. Returns (text, [(old, new), ...])."""
        a_ws, a_tok, ws = [], [], ""
        for t in _TOKEN.findall(original):
            if t.isspace():
                ws += t
            else:
                a_ws.append(ws)
                a_tok.append(t)
                ws = ""
        trailing = ws
        b_tok = [t for t in _TOKEN.findall(proposed) if not t.isspace()]
        sm = difflib.SequenceMatcher(None, [t.lower() for t in a_tok],
                                    [t.lower() for t in b_tok], autojunk=False)
        out, changes = [], []

        def keep(i1, i2):
            out.extend(a_ws[i] + a_tok[i] for i in range(i1, i2))

        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                keep(i1, i2)
                continue
            span, new = a_tok[i1:i2], b_tok[j1:j2]
            words = [t for t in span if t[0].isalpha()]
            new_words = [t for t in new if t[0].isalpha()]
            ok = (
                op == "replace"
                and bool(words) and all(not self.known(t) for t in words)
                and len(new_words) <= MAX_WORDS_PER_FIX * len(words)
                and not any(ch.isdigit() for t in new for ch in t)
                and [t for t in span if not t[0].isalpha()] == [t for t in new if not t[0].isalpha()]
            )
            if ok:      # the replacement must still sound like what was heard
                ok = _ratio("".join(words).lower(), "".join(new_words).lower()) >= MIN_SIMILARITY
            if ok:
                fixed = _join(new)
                if span[0][0].isupper() and fixed[0].islower():
                    fixed = fixed[0].upper() + fixed[1:]
                out.append(a_ws[i1] + fixed)
                changes.append((_join(span), fixed))
            elif len(words) == 1 and (fix := self.dictionary_fix(words[0])):
                # LLM edit rejected: swap just the one non-word, keep the rest
                out.append(a_ws[i1] + _join([fix if t == words[0] else t for t in span]))
                changes.append((words[0], fix))
            else:
                keep(i1, i2)
        return "".join(out) + trailing, changes

    def dictionary_only(self, text: str):
        """Fallback when the LLM isn't available: fix only near-certain non-words."""
        out, changes = [], []
        for t in _TOKEN.findall(text):
            fix = self.dictionary_fix(t) if (t[0].isalpha() and not self.known(t)) else None
            if fix:
                changes.append((t, fix))
            out.append(fix or t)
        return "".join(out), changes

    def correct(self, text: str):
        """Returns (corrected_text, [(old, new), ...]). Never raises."""
        if not text.strip():
            return text, []
        unknown = self.unknown_words(text)
        if not unknown:
            return text, []
        if not self.ready and not (self.proc and self.wait_ready(timeout=0.1)):
            return self.dictionary_only(text)
        try:
            proposed = self.ask_llm(text, unknown)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
            self.log(f"\n    corrector: LLM skipped ({type(e).__name__}), dictionary only\n")
            return self.dictionary_only(text)
        if not proposed:
            return self.dictionary_only(text)
        return self.guard(text, proposed)


if __name__ == "__main__":
    # Manual test: python corrector.py "some text with mispelled wrods"
    c = Corrector()
    c.start()
    if not c.wait_ready():
        sys.exit("llama-server did not become ready")
    for arg in sys.argv[1:] or ["This is a test of the dictation sofwhere from a medical standpoint."]:
        t0 = time.time()
        fixed, changes = c.correct(arg)
        print(f"\n{arg}\n-> {fixed}\n   {changes}  ({time.time() - t0:.1f}s)")
    c.stop()
