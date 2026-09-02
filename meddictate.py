#!/usr/bin/env python3
"""
meddictate.py - fully offline medical dictation with Google MedASR + sherpa-onnx.

Press F8 to start recording, F8 again to stop. Esc while recording cancels.

Two modes:
  LIVE (default)   Text appears phrase-by-phrase as you pause. The window and
                   text box you were in when you pressed F8 is LOCKED: you can
                   click into other windows to look things up, and the text
                   keeps landing in the locked box. Each phrase is appended at
                   the end of the box (Ctrl+End first), so you can fix a typo
                   mid-note without the next phrase being inserted at your
                   cursor. Focus is returned to wherever you were.
  BATCH (--batch)  Nothing is pasted until you press F8 to stop; then the whole
                   transcript is pasted into the locked box at once.

Nothing leaves this computer. No network access is used after setup.

Usage:
    python meddictate.py                 # live dictation
    python meddictate.py --batch         # batch dictation
    python meddictate.py --file x.wav    # transcribe a wav file (install test)
    python meddictate.py --hotkey f9     # use a different hotkey
    python meddictate.py --no-paste      # print instead of pasting (debugging)
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION - edit these to taste
# ----------------------------------------------------------------------------

HERE = Path(__file__).parent
MODEL_DIR = HERE / "sherpa-onnx-medasr-ctc-en-int8-2025-12-25"
VAD_MODEL = HERE / "silero_vad.onnx"
SAMPLE_RATE = 16000
NUM_THREADS = 6            # raise to your physical core count for faster decoding

# Live-mode phrase detection
PAUSE_SECONDS = 0.7        # silence this long ends a phrase and triggers a paste
MAX_PHRASE_SECONDS = 20    # force a split if you talk longer than this without pausing
PRE_PAD_SECONDS = 0.50     # audio kept before a detected phrase (the VAD reacts ~0.3s late)
POST_PAD_SECONDS = 0.20    # audio kept after a detected phrase

# Where text lands
PASTE_AT_END = False       # False: insert wherever the cursor is (a trailing space is
                           # added so text inserted before existing text isn't glued on).
                           # True: press Ctrl+End first so text always appends at the
                           # end of the box even if you moved the cursor to fix a typo.
                           # The widget has a Cursor/End toggle for this.
RETURN_TO_PREVIOUS_WINDOW = True   # after pasting, switch back to whatever window
                                   # you were looking at (e.g. UpToDate)

# Punctuation source. MedASR both guesses punctuation on its own (emitted as
# literal ".", ",", ...) and emits {period}, {comma}, ... tokens when you say
# them. With AUTO_PUNCTUATION = False only the punctuation you dictate is kept;
# the model's guesses are stripped (decimals like 2.5 are preserved) and words
# it capitalized after a guessed full stop are lowercased again, so sentence
# capitals follow your dictated punctuation only.
AUTO_PUNCTUATION = False

# Spoken-punctuation tokens MedASR emits when you dictate them out loud.
SPOKEN_PUNCTUATION = {
    "{period}": ".",
    "{full stop}": ".",
    "{comma}": ",",
    "{colon}": ":",
    "{semicolon}": ";",
    "{question mark}": "?",
    "{exclamation point}": "!",
    "{hyphen}": "-",
    "{dash}": " - ",
    "{slash}": "/",
    "{open paren}": "(",
    "{close paren}": ")",
    "{new line}": "\n",
    "{newline}": "\n",
    "{new paragraph}": "\n\n",
    "{next paragraph}": "\n\n",
}

# Spoken editing commands. Each occurrence acts once, so "backspace backspace"
# deletes two characters and "delete that delete that" two words. They first
# eat text in the phrase being spoken; anything left over is sent to the note
# as real Backspace keystrokes against what was pasted earlier this session.
EDIT_COMMANDS = {
    "backspace": "backspace",        # delete one character
    "back space": "backspace",
    "delete that": "delete_word",    # delete the previous whole word
}

# MedASR marks spoken commands with {braces}, but for commands it wasn't trained
# on it garbles the contents ({nexcktpace} for "backspace"). Any unknown {token}
# is snapped to the most similar known punctuation/edit command if the
# similarity (0..1) is at least this; otherwise it is dropped.
FUZZY_TOKEN_MATCH = 0.55

# Turn recognised section names ("findings", "impression") into "FINDINGS:"
# header lines. False = leave them as ordinary words; dictate layout yourself.
SECTION_HEADERS = False

# Optional spelling-repair stage (corrector.py): a local MedGemma model fixes
# non-words, but only words absent from the English/medical dictionaries may
# change. Needs llama\ and models\ next to this script; silently skipped if not.
USE_CORRECTOR = True

# Show the model's raw output for each phrase in the log (before cleanup), as
# "raw: ..." lines. Useful to see how a spoken command was actually heard.
SHOW_RAW = True

# Filler words removed from the transcript wherever they appear (whole words,
# case-insensitive). "er" is deliberately absent so "ER" survives.
FILLER_WORDS = {"um", "umm", "uhm", "uhmm", "uh", "uhh", "hmm", "hm", "mhm", "mm", "ah", "erm"}

# Your personal correction dictionary. Add drug names, colleague names,
# local hospitals, etc. that the model consistently gets wrong.
# Matching is case-insensitive on the left; right side is pasted verbatim.
CORRECTIONS = {
    # "lice in a pril": "lisinopril",
    # "hartford hospital": "Hartford Hospital",
}

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------


def load_recognizer():
    import sherpa_onnx

    model = MODEL_DIR / "model.int8.onnx"
    tokens = MODEL_DIR / "tokens.txt"
    if not model.is_file() or not tokens.is_file():
        sys.exit(
            f"Model not found in {MODEL_DIR}\n"
            "Download it with:\n"
            "  curl.exe -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2\n"
            "  tar -xf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2"
        )
    return sherpa_onnx.OfflineRecognizer.from_medasr_ctc(
        model=str(model), tokens=str(tokens), num_threads=NUM_THREADS,
    )


class PhraseSplitter:
    """Wraps the silero VAD and returns phrases padded with a little audio on
    each side, so the first and last syllables aren't clipped."""

    def __init__(self):
        import sherpa_onnx

        if not VAD_MODEL.is_file():
            sys.exit(
                f"VAD model not found: {VAD_MODEL}\n"
                "Download it with:\n"
                "  curl.exe -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "asr-models/silero_vad.onnx"
            )
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(VAD_MODEL)
        cfg.silero_vad.min_silence_duration = PAUSE_SECONDS
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = MAX_PHRASE_SECONDS
        cfg.sample_rate = SAMPLE_RATE
        self.window = cfg.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.pending = np.zeros(0, dtype=np.float32)
        self.fed = []          # everything handed to the VAD this session
        self.fed_len = 0

    def _drain(self, final=False):
        pre = int(PRE_PAD_SECONDS * SAMPLE_RATE)
        post = int(POST_PAD_SECONDS * SAMPLE_RATE)
        out = []
        while not self.vad.empty():
            seg = self.vad.front
            s, e = seg.start, seg.start + len(seg.samples)
            self.vad.pop()
            if not self.fed:
                out.append(np.asarray(seg.samples, dtype=np.float32))
                continue
            audio = np.concatenate(self.fed)
            self.fed = [audio]
            out.append(audio[max(0, s - pre): min(len(audio), e + post)])
        return out

    def feed(self, samples):
        """Feed mic audio; returns a list of completed phrases (np arrays)."""
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= self.window:
            chunk = self.pending[:self.window]
            self.pending = self.pending[self.window:]
            self.vad.accept_waveform(chunk)
            self.fed.append(chunk)
            self.fed_len += len(chunk)
        return self._drain()

    def flush(self):
        """Call when recording stops; returns any trailing phrase."""
        if len(self.pending):
            pad = np.zeros(self.window - len(self.pending), dtype=np.float32)
            chunk = np.concatenate([self.pending, pad])
            self.vad.accept_waveform(chunk)
            self.fed.append(chunk)
            self.pending = np.zeros(0, dtype=np.float32)
        self.vad.flush()
        return self._drain(final=True)


def transcribe(recognizer, samples: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    recognizer.decode_stream(stream)
    return stream.result.text


# ----------------------------------------------------------------------------
# Text cleanup
# ----------------------------------------------------------------------------


def strip_model_punctuation(text: str) -> str:
    """Remove punctuation the model guessed, keeping {spoken} tokens for later.

    A plain Capitalized word right after a guessed full stop is lowercased
    (all-caps abbreviations such as PE or CT are left as they are). Punctuation
    between two digits (2.5, 1,000) is kept.
    """
    t = re.sub(r"[.?!]+(\s+)([A-Z])(?=[a-z])",
               lambda m: m.group(1) + m.group(2).lower(), text)
    t = re.sub(r"(?<!\d)[.,:;?!]|[.,:;?!](?!\d)", "", t)
    return t


_HDR = "\x01"   # placeholder for a section header's own line break, resolved last


def clean_text(text: str, continuing: bool = False, prev: str = "") -> str:
    """Convert MedASR's spoken-punctuation tokens and apply corrections.

    continuing=True means this phrase follows text that did not end a
    sentence, so the first word should not be capitalized. prev is what has
    already been written this session; it decides whether a section header at
    the start of the phrase needs a line break of its own.

    Dictated line breaks are kept exactly: "new line" is one Enter, "new
    paragraph" is two, and repeated commands add up.
    """
    t = text

    if not AUTO_PUNCTUATION:
        t = strip_model_punctuation(t)
        if continuing:
            # model capitalized the phrase start on its own; we're mid-sentence
            t = re.sub(r"^(\s*)([A-Z])(?=[a-z])", lambda m: m.group(1) + m.group(2).lower(), t)

    for token, replacement in SPOKEN_PUNCTUATION.items():
        t = re.sub(re.escape(token), replacement, t, flags=re.IGNORECASE)
    t = re.sub(r"\{[^}]*\}", "", t)                       # drop unknown {tokens}

    # MedASR tags section names it recognises, e.g. [FINDINGS]. With
    # SECTION_HEADERS they become "FINDINGS:" on their own line; otherwise they
    # are just the words you said, inline ("findings"), and any layout is yours
    # to dictate.
    if SECTION_HEADERS:
        t = re.sub(r"[ \t]*\[([A-Z][A-Z /]+)\][ \t]*:?[ \t]*",
                   lambda m: f"{_HDR}{m.group(1).strip()}: ", t)
    else:
        # the model capitalizes the word after a tag as a sentence start; undo that
        t = re.sub(r"\[([A-Z][A-Z /]+)\]\s*([A-Z])(?=[a-z])",
                   lambda m: f" {m.group(1).strip().lower()} {m.group(2).lower()}", t)
        t = re.sub(r"\[([A-Z][A-Z /]+)\]", lambda m: f" {m.group(1).strip().lower()} ", t)

    # Anything else containing a bracket is a de-identification tag the model
    # learned from redacted training notes ([ProfessionalName], [Date]) or a
    # fragment of one ("e]", "essionalName]"), emitted spuriously around pauses.
    # Drop the whole chunk; mention complete tags so a swallowed name is noticed.
    tags = re.findall(r"\[[A-Za-z][A-Za-z ]*\]", t)
    if tags:
        print(f"\n    (dropped model tag: {' '.join(tags)})\n", end="", flush=True)
    t = re.sub(r"\S*[\[\]]\S*", " ", t)

    if FILLER_WORDS:
        filler = "|".join(sorted(map(re.escape, FILLER_WORDS), key=len, reverse=True))
        t = re.sub(rf"(?<![A-Za-z'])(?:{filler})(?![A-Za-z'])", " ", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+([.,:;?!)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)

    # A section header starts its own line unless a line break is already
    # there (dictated just before it, or the previous phrase ended with one).
    src = t

    def header_break(m):
        before = src[:m.start()].rstrip(" ")
        if before == "":
            return "\n" if (prev and not prev.endswith("\n")) else ""
        return "" if before.endswith("\n") else "\n"

    t = re.sub(_HDR, header_break, t)

    # Capitalize after sentence-ending punctuation / colons / newlines
    t = re.sub(r"([.?!:]\s+|\n\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    t = t.strip(" \t")
    if t and not continuing:
        t = t[0].upper() + t[1:]

    for wrong, right in CORRECTIONS.items():
        t = re.sub(re.escape(wrong), right, t, flags=re.IGNORECASE)
    return t


def ends_sentence(text: str) -> bool:
    text = text.rstrip(" \t")
    return bool(text) and text[-1] in ".?!:\n"


# ----------------------------------------------------------------------------
# Window lock (Windows only; harmless no-op elsewhere)
# ----------------------------------------------------------------------------

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    class GUITHREADINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
            ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
            ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
            ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
            ("rcCaret", wintypes.RECT),
        ]

    def _thread_of(hwnd):
        return user32.GetWindowThreadProcessId(hwnd, None)

    def _focused_control(hwnd):
        gti = GUITHREADINFO()
        gti.cbSize = ctypes.sizeof(GUITHREADINFO)
        if user32.GetGUIThreadInfo(_thread_of(hwnd), ctypes.byref(gti)):
            return gti.hwndFocus
        return None

    def _window_title(hwnd):
        n = user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(n)
        user32.GetWindowTextW(hwnd, buf, n)
        return buf.value


class Target:
    """The window + text box that was active when recording started."""

    def __init__(self):
        self.hwnd = None
        self.control = None
        self.title = ""
        if IS_WINDOWS:
            self.hwnd = user32.GetForegroundWindow()
            self.control = _focused_control(self.hwnd)
            self.title = _window_title(self.hwnd)

    def focus(self) -> bool:
        """Bring the locked window/box to the front. Returns True on success."""
        if not IS_WINDOWS or not self.hwnd:
            return True
        if not user32.IsWindow(self.hwnd):
            print("  (locked window no longer exists; pasting to current window)")
            return False
        already = (user32.GetForegroundWindow() == self.hwnd)
        me = kernel32.GetCurrentThreadId()
        fg = _thread_of(user32.GetForegroundWindow())
        tgt = _thread_of(self.hwnd)
        user32.AttachThreadInput(me, fg, True)
        user32.AttachThreadInput(me, tgt, True)
        try:
            if user32.IsIconic(self.hwnd):
                user32.ShowWindow(self.hwnd, 9)          # SW_RESTORE
            if not already:
                if not user32.SetForegroundWindow(self.hwnd):
                    # Windows sometimes refuses focus changes; tapping Alt unlocks it
                    import keyboard
                    keyboard.press_and_release("alt")
                    user32.SetForegroundWindow(self.hwnd)
            if self.control and user32.IsWindow(self.control):
                user32.SetFocus(self.control)
        finally:
            user32.AttachThreadInput(me, fg, False)
            user32.AttachThreadInput(me, tgt, False)
        if not already:
            time.sleep(0.15)
        return True


def current_window():
    if IS_WINDOWS:
        return user32.GetForegroundWindow()
    return None


def restore_window(hwnd):
    if IS_WINDOWS and hwnd and user32.IsWindow(hwnd):
        me = kernel32.GetCurrentThreadId()
        tgt = _thread_of(hwnd)
        user32.AttachThreadInput(me, tgt, True)
        user32.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(me, tgt, False)


# ----------------------------------------------------------------------------
# Pasting
# ----------------------------------------------------------------------------


def _in_target(target: Target, at_end: bool, action):
    """Focus the locked box, optionally jump to its end, run action, come back."""
    import pyautogui

    came_from = current_window()
    target.focus()
    if at_end:
        pyautogui.hotkey("ctrl", "end")
        time.sleep(0.03)
    action()
    if RETURN_TO_PREVIOUS_WINDOW and came_from and came_from != target.hwnd:
        restore_window(came_from)


def paste_text(text: str, target: Target, at_end: bool):
    import pyperclip
    import pyautogui

    def do_paste():
        old_clip = None
        try:
            old_clip = pyperclip.paste()
        except Exception:
            pass
        pyperclip.copy(text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.25)
        if old_clip is not None:
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

    _in_target(target, at_end, do_paste)


def press_backspace(count: int, target: Target, at_end: bool, word: bool = False):
    """Send Backspace (or Ctrl+Backspace for a word) count times to the locked box."""
    import pyautogui

    def do_press():
        for _ in range(count):
            if word:
                pyautogui.hotkey("ctrl", "backspace")
            else:
                pyautogui.press("backspace")
            time.sleep(0.02)
        time.sleep(0.1)

    _in_target(target, at_end, do_press)


# ----------------------------------------------------------------------------
# Dictation loop
# ----------------------------------------------------------------------------


def normalize_tokens(raw: str) -> str:
    """Snap garbled {command} tokens to the closest known one."""
    import difflib

    known = [k.strip("{}") for k in SPOKEN_PUNCTUATION] + list(EDIT_COMMANDS)

    def fix(m):
        content = re.sub(r"\s+", " ", m.group(1).strip().lower())
        if not content or content in known:
            return m.group(0)
        best = max(known, key=lambda k: difflib.SequenceMatcher(None, content, k).ratio())
        if difflib.SequenceMatcher(None, content, best).ratio() >= FUZZY_TOKEN_MATCH:
            return "{" + best + "}"
        return m.group(0)

    return re.sub(r"\{([^{}]*)\}", fix, raw)


def split_commands(raw: str):
    """Split raw model text into ("text", str) and (command, count) segments.

    Commands are matched as whole words, with or without the model's {braces};
    consecutive repeats are merged into one segment with a count.
    """
    if not EDIT_COMMANDS:
        return [("text", raw)]
    names = sorted(EDIT_COMMANDS, key=len, reverse=True)
    alts = "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in names)
    pat = re.compile(rf"(?<![A-Za-z])\{{?\s*({alts})\s*\}}?(?![A-Za-z])", re.IGNORECASE)
    segs, pos = [], 0
    for m in pat.finditer(raw):
        if raw[pos:m.start()].strip():
            segs.append(("text", raw[pos:m.start()]))
        kind = EDIT_COMMANDS[re.sub(r"\s+", " ", m.group(1).lower())]
        if segs and segs[-1][0] == kind:
            segs[-1] = (kind, segs[-1][1] + 1)
        else:
            segs.append((kind, 1))
        pos = m.end()
    if raw[pos:].strip():
        segs.append(("text", raw[pos:]))
    return segs


def drop_last_word(s: str) -> str:
    """Remove the final word (with the space before it). A run of trailing
    line breaks counts as one 'word' so 'delete that' can undo a new paragraph."""
    s = s.rstrip(" ")
    if s.endswith("\n"):
        return s.rstrip("\n")
    return re.sub(r" *\S+$", "", s)


class PhraseWriter:
    """Joins live phrases with correct spacing/capitalization, applies spoken
    editing commands, and pastes the result. self.doc mirrors everything this
    session has put into the locked box so deletions can reach back into it."""

    def __init__(self, target, do_paste):
        self.target = target
        self.do_paste = do_paste
        self.doc = ""

    def emit(self, raw: str):
        if SHOW_RAW:
            print(f"\n    raw: {raw.strip()!r}\n", end="", flush=True)
        raw = normalize_tokens(raw)
        pending = ""
        for kind, val in split_commands(raw):
            if kind == "text":
                before = self.doc + pending
                continuing = bool(before) and not ends_sentence(before)
                text = clean_text(val, continuing=continuing, prev=before)
                # a lone repeated punctuation mark ("period" heard twice across a
                # pause) would give ".." - drop the duplicate
                while text and text[0] in ".,:;?!" and before.rstrip().endswith(text[0]):
                    text = text[1:].lstrip(" ")
                if not text:
                    continue
                if text[0] in ".,:;?!)" and not pending and self.doc.endswith(" "):
                    # punctuation arriving on its own after our trailing space
                    # (cursor mode): pull the space back so we get "word." not "word ."
                    self._edit_doc("backspace", 1, quiet=True)
                    before = self.doc
                if (before and not before.endswith(("\n", " "))
                        and not text.startswith("\n") and text[0] not in ".,:;?!)"):
                    text = " " + text
                pending += text
                continue
            # editing command: consume the current phrase first, then the note
            while val and pending:
                pending = pending[:-1] if kind == "backspace" else drop_last_word(pending)
                val -= 1
            if val:
                self._edit_doc(kind, val)
        if pending:
            if corrector is not None:
                pending, changes = corrector.correct(pending)
                if changes:
                    print("\n    fixed: " + ", ".join(f"{a} -> {b}" for a, b in changes) + "\n",
                          end="", flush=True)
            if not PASTE_AT_END and not pending.endswith(("\n", " ")):
                pending += " "      # cursor mode: don't glue onto whatever follows
            print(pending, end="", flush=True)
            if self.do_paste:
                paste_text(pending, self.target, PASTE_AT_END)
            self.doc += pending

    def _edit_doc(self, kind: str, count: int, quiet: bool = False):
        if kind == "backspace":
            if not quiet:
                print(f"[⌫x{count}]", end="", flush=True)
            self.doc = self.doc[:-count]
            if self.do_paste:
                press_backspace(count, self.target, PASTE_AT_END)
            return
        # delete_word: we know exactly what we pasted, so delete that many
        # characters; fall back to Ctrl+Backspace for text we didn't write
        print(f"[delete word x{count}]", end="", flush=True)
        keys = 0
        for _ in range(count):
            if not self.doc:
                if self.do_paste:
                    press_backspace(1, self.target, PASTE_AT_END, word=True)
                continue
            new = drop_last_word(self.doc)
            keys += len(self.doc) - len(new)
            self.doc = new
        if keys and self.do_paste:
            press_backspace(keys, self.target, PASTE_AT_END)


def record_until_hotkey(hotkey, on_audio):
    """Stream mic audio to on_audio(samples) until hotkey/Esc. Returns cancelled."""
    import keyboard
    import sounddevice as sd

    cancelled = False
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as s:
        while True:
            samples, _ = s.read(int(0.1 * SAMPLE_RATE))
            on_audio(samples.reshape(-1))
            if keyboard.is_pressed(hotkey):
                break
            if keyboard.is_pressed("esc"):
                cancelled = True
                break
    return cancelled


# The optional spelling-repair stage; set by start_corrector().
corrector = None


def start_corrector():
    """Start the LLM spelling stage in the background. Returns it, or None if
    it isn't installed. Phrases spoken before it is ready are pasted as-is."""
    import threading

    if not USE_CORRECTOR:
        return None
    try:
        from corrector import Corrector
        c = Corrector(log=print)
        c.start()
    except Exception as e:
        print(f"Corrector unavailable: {e}")
        return None
    if not c.proc:
        return None
    threading.Thread(target=c.wait_ready, daemon=True).start()
    return c


# Status hook for a UI wrapper (meddictate_ui.py). Called with one of
# "idle", "recording", "busy" and an optional detail string. No-op by default.
def on_event(event: str, detail: str = ""):
    pass


def wait_for_hotkey(get_hotkey):
    """Block until the hotkey is pressed. get_hotkey() is re-read on every
    poll so a UI can change the key while we are waiting. Waits for the key to
    be released first, so the press that stopped a recording can't restart it."""
    import keyboard

    while keyboard.is_pressed(get_hotkey()):
        time.sleep(0.03)
    while not keyboard.is_pressed(get_hotkey()):
        time.sleep(0.03)


def run_dictation(recognizer, hotkey: str, do_paste: bool, live: bool, settings=None):
    """settings, if given, is a dict {"hotkey": str, "live": bool} that the
    caller may change at any time; it is re-read each time recording starts."""
    if settings is None:
        settings = {"hotkey": hotkey, "live": live}

    splitter = PhraseSplitter()
    mode = "LIVE" if settings["live"] else "BATCH"
    hk = settings["hotkey"].upper()
    print(f"\n[{mode}] Ready. Click into your note, press {hk} to start, "
          f"{hk} again to stop, Esc to cancel.")
    print("You may click into other windows while dictating; text stays in the locked box.")
    print("Ctrl+C in this window to quit.\n")

    while True:
        on_event("idle")
        wait_for_hotkey(lambda: settings["hotkey"])
        hotkey, live = settings["hotkey"], settings["live"]
        time.sleep(0.25)
        target = Target()
        on_event("recording", target.title or "current window")
        print("\a", end="")
        print(f"● Recording -> locked to: {target.title or 'current window'}", flush=True)

        writer = PhraseWriter(target, do_paste)
        all_chunks = []

        def on_audio(samples):
            if not live:
                all_chunks.append(samples.copy())
                return
            for phrase in splitter.feed(samples):
                writer.emit(transcribe(recognizer, phrase, SAMPLE_RATE))

        cancelled = record_until_hotkey(hotkey, on_audio)
        on_event("busy")
        time.sleep(0.25)

        if live:
            if not cancelled:
                for phrase in splitter.flush():
                    writer.emit(transcribe(recognizer, phrase, SAMPLE_RATE))
            splitter.reset()
            print("\n" + ("✗ Cancelled" if cancelled else "■ Stopped") + "\n")
            continue

        # batch mode
        if cancelled or not all_chunks:
            print("✗ Cancelled\n")
            continue
        samples = np.concatenate(all_chunks)
        seconds = len(samples) / SAMPLE_RATE
        if seconds < 0.5:
            print("✗ Too short, ignored\n")
            continue
        t0 = time.time()
        batch = PhraseWriter(target, do_paste=False)
        batch.emit(transcribe(recognizer, samples, SAMPLE_RATE))
        text = batch.doc
        print(f"\n✓ {seconds:.1f}s of audio transcribed in {time.time() - t0:.1f}s\n")
        if do_paste and text:
            paste_text(text, target, PASTE_AT_END)


# ----------------------------------------------------------------------------
# File mode (install test)
# ----------------------------------------------------------------------------


def transcribe_file(recognizer, path: str, live: bool) -> str:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio[:, 0])
    if sr != SAMPLE_RATE:
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    if not live:
        batch = PhraseWriter(Target(), do_paste=False)
        batch.emit(transcribe(recognizer, audio, SAMPLE_RATE))
        return ""

    # Simulate live mode: run the file through the VAD phrase splitter
    splitter = PhraseSplitter()
    writer = PhraseWriter(Target(), do_paste=False)
    step = int(0.1 * SAMPLE_RATE)
    for pos in range(0, len(audio), step):
        for phrase in splitter.feed(audio[pos:pos + step]):
            writer.emit(transcribe(recognizer, phrase, SAMPLE_RATE))
    for phrase in splitter.flush():
        writer.emit(transcribe(recognizer, phrase, SAMPLE_RATE))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="transcribe a wav file and exit")
    ap.add_argument("--batch", action="store_true", help="paste everything at the end instead of live")
    ap.add_argument("--hotkey", default="f8", help="start/stop key (default f8)")
    ap.add_argument("--no-paste", action="store_true", help="print only, don't paste")
    args = ap.parse_args()

    global corrector
    corrector = start_corrector()

    print("Loading MedASR...", end=" ", flush=True)
    t0 = time.time()
    recognizer = load_recognizer()
    print(f"done ({time.time() - t0:.1f}s)")

    try:
        if args.file:
            if corrector is not None:
                corrector.wait_ready()
            out = transcribe_file(recognizer, args.file, live=not args.batch)
            if out:
                print(out)
            print()
            return

        run_dictation(recognizer, args.hotkey, not args.no_paste, live=not args.batch)
    finally:
        if corrector is not None:
            corrector.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
