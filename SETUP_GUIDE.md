# MedASR Offline Dictation — Install Brief for Claude Code

> **Jon, read this box; everything below it is for the agent.**
>
> 1. Put this file, `meddictate.py`, and `start_dictation.bat` together in a new folder, e.g. `C:\Users\<you>\meddictate`.
> 2. Open Claude Desktop → **Code** tab (or run `claude` in a terminal) with that folder as the working directory. **Use Code, not Cowork.** Cowork runs in a sandboxed VM and cannot install Python on your actual PC, use your microphone, or register a hotkey.
> 3. Paste this as your first message: *"Read SETUP_GUIDE.md in this folder and carry out the install brief. Stop and hand off to me at the point it says to."*
> 4. Approve the commands it proposes. It will finish with a file test, then hand the live-microphone check to you (it can't press F8 or grant mic permission on your behalf).
>
> Budget: ~10 minutes, ~300 MB of downloads, $0.

---

## Objective

Install a fully offline medical dictation system on this Windows PC: Google's MedASR speech model running through sherpa-onnx, driven by the `meddictate.py` script already in this folder. When done, double-clicking `start_dictation.bat` gives push-to-talk (F8) dictation that pastes into whatever text box the user was in.

## Hard constraints

- **Nothing about this setup may send audio or text off the machine.** Do not install or configure any cloud speech service, API key, or telemetry. The only network use is downloading Python, pip packages, and two model files from the URLs below.
- **Do not edit the logic of `meddictate.py`.** The one permitted unprompted edit is `NUM_THREADS` (Step 5). You may fix a path or a Windows-specific import if something fails, but tell the user exactly what you changed and why. If you think the script has a bug, explain it and ask before changing behaviour.
- Use the Python 3.12 line, not the newest release; sherpa-onnx ships prebuilt Windows wheels for 3.12.
- Work in this folder only. Don't install anything system-wide beyond Python itself.
- Ask before anything that needs administrator rights.

## Files that should already be here

| File | Purpose |
|---|---|
| `SETUP_GUIDE.md` | This brief |
| `meddictate.py` | The dictation script (hotkey, mic, phrase splitting, window lock, paste) |
| `start_dictation.bat` | Double-click launcher |

If either of the last two is missing, stop and tell the user; do not recreate them from memory.

---

## Step 1 — Python 3.12

Check first:

```powershell
py -3.12 --version
```

If that prints `Python 3.12.x`, skip to Step 2. Otherwise install it:

```powershell
winget install --id Python.Python.3.12 -e --source winget --accept-source-agreements --accept-package-agreements
```

If `winget` is unavailable, direct the user to https://www.python.org/downloads/release/python-3120/ → "Windows installer (64-bit)", and tell them to tick **"Add python.exe to PATH"**. After install, **open a fresh shell** (PATH changes don't apply to the current one) and re-run the version check. Use the `py -3.12` launcher throughout rather than bare `python`, so a second Python on the machine can't interfere.

**Verify:** `py -3.12 --version` → `Python 3.12.x`.

## Step 2 — Python packages

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install sherpa-onnx numpy soundfile sounddevice keyboard pyperclip pyautogui
```

**Verify:**

```powershell
py -3.12 -c "import sherpa_onnx, sounddevice, keyboard, pyperclip, pyautogui, soundfile; print('imports ok')"
```

If `sherpa-onnx` has no wheel for this Python/architecture, report the exact pip error and stop; do not try to build from source.

## Step 3 — Model files (into this folder)

```powershell
curl.exe -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
tar -xf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
del sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
curl.exe -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

If `tar` can't handle bzip2 on this machine, use `py -3.12 -c "import tarfile; tarfile.open('sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2').extractall()"` instead.

**Verify** the folder now contains:

```
meddictate.py
start_dictation.bat
silero_vad.onnx                                   (~630 KB)
sherpa-onnx-medasr-ctc-en-int8-2025-12-25\
    model.int8.onnx                               (~154 MB)
    tokens.txt
    test_wavs\0.wav ... 5.wav
```

Check `model.int8.onnx` is roughly 154 MB; a tiny file means a failed/HTML download.

## Step 4 — Launcher

Confirm `start_dictation.bat` contains `py -3.12 meddictate.py %*` (not bare `python`). Fix it if not.

## Step 5 — File-mode test (no microphone needed)

```powershell
py -3.12 meddictate.py --file sherpa-onnx-medasr-ctc-en-int8-2025-12-25\test_wavs\0.wav
```

**Expected output** (a sample radiology dictation; it runs through the same phrase splitter live mode uses):

```
Loading MedASR... done (Ns)
EXAM TYPE: CT chest PE protocol.
INDICATION: 54-year-old female, shortness of breath, evaluate for PE.
TECHNIQUE: Standard protocol.
FINDINGS: Pulmonary vasculature: The main PA is patent. There are filling defects in the segmental branches of the right lower lobe, compatible with acute PE. No saddle embolus. Lungs: No pneumothorax. Small bilateral effusions, right greater than left.
IMPRESSION: Acute segmental PE right lower lobe.
```

Minor wording differences are fine; the structure and medical terms should match. Also run `--file ...\test_wavs\4.wav` and confirm it begins "Please take your medication as prescribed by your doctor". If either output is empty, garbled, or errors, stop and report the full console output.

Then set the thread count: run `py -3.12 -c "import os; print(os.cpu_count())"` and in `meddictate.py` set `NUM_THREADS` to half that number (physical cores), minimum 2.

## Step 6 — Optional conveniences (ask the user first)

- **Auto-start with Windows:** a shortcut to `start_dictation.bat` in the folder that opens from `explorer shell:startup`.
- **Desktop shortcut** to `start_dictation.bat`.

## Step 7 — HAND OFF TO THE USER

Stop here. Print the following for the user, then end your turn:

> Install complete and file test passed. The live-microphone test is yours to do:
>
> 1. Settings → Privacy & security → Microphone → make sure "Let desktop apps access your microphone" is **On**.
> 2. Double-click `start_dictation.bat`. It should say `[LIVE] Ready`.
> 3. Click into Notepad, press **F8**, say a sentence, pause, say another, press **F8** again. Text should appear in Notepad phrase by phrase.
> 4. Then test the lock: press F8, start talking, click into a browser window, keep talking. Text should keep landing in Notepad and focus should flip back to the browser after each phrase.
>
> If anything misbehaves, copy what the console window printed and tell me.

You cannot do these steps yourself: they need the microphone and a human pressing a hotkey, and launching the script from an agent session would hold the mic and hotkey. **Do not run `meddictate.py` without `--file`.**

---

## Reference for the user (after install)

### How to dictate

- **Start/stop:** F8. **Cancel:** Esc while recording.
- **Live mode** (default): each phrase is transcribed when you pause ~¾ s and pasted immediately. The note box you were in at F8 is *locked* for the session: click into UpToDate or a chart and keep talking, and each phrase pastes into the locked box, then focus returns to where you were. Fix a typo mid-note freely; each new phrase does Ctrl+End first, so it appends at the end rather than at your cursor.
- **Batch mode:** `start_dictation.bat --batch` (or edit the .bat). Nothing appears until you press F8 to stop; then the whole transcript is pasted into the locked box.
- **Punctuation:** just talk and it auto-punctuates, or dictate "period", "comma", "colon", "new paragraph" Dragon-style. Saying a section name like "findings" or "impression" on its own makes a header line.

### Tuning (top of `meddictate.py`, open in Notepad)

| Setting | Default | Change when |
|---|---|---|
| `NUM_THREADS` | 4 | Physical core count (the agent sets this) |
| `PAUSE_SECONDS` | 0.7 | Phrases split mid-sentence → raise to 1.0; output feels laggy → lower |
| `PRE_PAD_SECONDS` | 0.5 | First syllable of phrases clipped → raise; a word duplicated across phrases → lower |
| `PASTE_AT_END` | True | Set False to insert at the cursor (e.g. dictating into the middle of a template) |
| `RETURN_TO_PREVIOUS_WINDOW` | True | Set False if the window flipping is distracting |
| `CORRECTIONS` | empty | Add `"what it heard": "what you meant",` for consistent mishearings (drug names, colleagues, hospitals) |

### Troubleshooting

| Symptom | Fix |
|---|---|
| Nothing pastes, transcript shows in console | Some apps (Citrix/RDP, admin windows) block synthetic keystrokes. Run the .bat as administrator; else use `--no-paste` and copy from the console |
| Text lands in the wrong window after clicking away | Windows sometimes refuses focus changes; the script taps Alt and retries. If it persists in a specific app, note the app name |
| Silent / empty transcripts | Wrong default mic: Settings → Sound → Input. Or list devices: `py -3.12 -c "import sounddevice; print(sounddevice.query_devices())"` |
| Odd `{tokens}` in output | Add the spoken word to `SPOKEN_PUNCTUATION` in the script |

### Licensing / "free forever"

sherpa-onnx, silero VAD and all Python packages are Apache/MIT/BSD. The MedASR weights are under Google's Health AI Developer Foundations terms (https://developers.google.com/health-ai-developer-foundations/terms): free to use and fine-tune locally, but not a strict OSI license, and worth a five-minute read for clinical use. Once the files are on disk nothing checks in with anyone; it works indefinitely with the network cable unplugged.
