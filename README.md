# meddictate — offline medical dictation for Windows

Push-to-talk clinical dictation that runs entirely on the local machine.
Google's **MedASR** speech model (via sherpa-onnx) transcribes; a small local
**MedGemma** model (via llama.cpp) repairs misspelled non-words under a strict
dictionary guard; the text is pasted into whatever note box you were in.
Nothing leaves the PC — no cloud speech, no API keys, no telemetry.

## Files

| File | Purpose |
|---|---|
| `meddictate.py` | Core: hotkey, mic, VAD phrase splitting, text cleanup, spoken commands, window lock, paste |
| `meddictate_ui.py` | Small always-on-top widget (status dot, Live/Batch toggle, hotkey picker, log). Preferred launcher |
| `corrector.py` | Optional LLM spelling repair with dictionary guard |
| `wordlists/custom_words.txt` | Your own words (brands, abbreviations) — never altered by the corrector |
| `start_dictation_widget.bat` | Launch the widget (no console window) |
| `start_dictation.bat` | Launch the plain console version (debugging) |
| `SETUP_GUIDE.md` | The original install brief; superseded in places by this README |

## Install

Python 3.12 (`winget install --id Python.Python.3.12 -e`), then:

```powershell
py -3.12 -m pip install sherpa-onnx numpy soundfile sounddevice keyboard pyperclip pyautogui
```

Speech models (into this folder):

```powershell
curl.exe -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
tar -xf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
del sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
curl.exe -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

Corrector (optional — skipped automatically if missing):

```powershell
# llama.cpp Windows build (Vulkan works on AMD and NVIDIA); unzip into .\llama\
curl.exe -L -o llama.zip https://github.com/ggml-org/llama.cpp/releases/download/b10752/llama-b10752-bin-win-vulkan-x64.zip
Expand-Archive llama.zip llama; del llama.zip
# MedGemma 4B (2.5 GB) into .\models\
mkdir models; curl.exe -L -o models\medgemma-4b-it-Q4_K_M.gguf https://huggingface.co/unsloth/medgemma-4b-it-GGUF/resolve/main/medgemma-4b-it-Q4_K_M.gguf
# word lists into .\wordlists\
curl.exe -L -o wordlists\english_words.txt https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
curl.exe -L -o wordlists\medical_terms.txt https://raw.githubusercontent.com/glutanimate/wordlist-medicalterms-en/master/wordlist.txt
```

Test without a microphone:

```powershell
py -3.12 meddictate.py --file sherpa-onnx-medasr-ctc-en-int8-2025-12-25\test_wavs\0.wav
```

Launch: `pyw -3.12 meddictate_ui.py` (or the `.bat`). A shortcut to that in
`shell:startup` makes it start with Windows.

## Dictating

- **Hotkey** (default F8, changeable in the widget) starts and stops; **Esc** cancels.
- **Live** mode pastes each phrase when you pause ~0.7 s; **Batch** pastes everything when you stop.
- The note box you were in when you pressed the hotkey is *locked*: you can click elsewhere and text keeps landing there.
- **Cursor / End** (widget toggle, `PASTE_AT_END`): insert at the cursor position (default; a trailing space is added so text placed in front of existing text isn't glued on), or always append at the end of the box.
- **Punctuation is dictated, never guessed**: "period", "comma", "colon", "question mark", "new line" (one Enter), "new paragraph" (two). Commands stack. The first letter of each sentence is capitalized automatically.
- **Editing**: "backspace" (one character), "delete that" (previous word); both stack and reach back into text already pasted.
- Fillers (um, uh, …) are removed. Section names ("impression", "findings") are ordinary words, not headers, unless `SECTION_HEADERS = True`.
- The widget's log shows the model's raw output (`raw:`) and every corrector change (`fixed: a -> b`) so you can audit it.

## The dictionary guard

The LLM may only change words that are **absent** from the English + medical +
custom word lists. Numbers, units, doses, punctuation, capitalization,
abbreviations and every dictionary word are pasted exactly as MedASR produced
them; multi-word expansions must still sound like the original. Real-word
substitutions (present/presents) are therefore *not* corrected — proofread for
those.

## Tuning

Top of `meddictate.py`: `NUM_THREADS`, `PAUSE_SECONDS`, `PRE_PAD_SECONDS`,
`AUTO_PUNCTUATION`, `SECTION_HEADERS`, `USE_CORRECTOR`, `SHOW_RAW`,
`SPOKEN_PUNCTUATION`, `EDIT_COMMANDS`, `FILLER_WORDS`, `CORRECTIONS`.
Top of `corrector.py`: similarity thresholds, `GPU_LAYERS`, timeouts.

## Licences

Code: MIT (this repository). sherpa-onnx, silero VAD and the Python packages
are Apache/MIT/BSD. MedASR and MedGemma weights: Google Health AI Developer
Foundations terms. Word lists: dwyl/english-words (Unlicense),
glutanimate/wordlist-medicalterms-en (GPLv3, downloaded separately).
