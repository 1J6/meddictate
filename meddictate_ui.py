#!/usr/bin/env python3
"""
meddictate_ui.py - small always-on-top status widget around meddictate.py.

The dictation loop runs unchanged in a background thread; this window shows
what it is doing and collects everything it would have printed to a console:

    grey dot    idle, waiting for the hotkey
    green dot   recording
    amber dot   loading the model / transcribing
    red dot     something went wrong (open the log)

Controls: Live/Batch mode toggle and a hotkey picker (both apply from the
next recording; disabled while recording). The v button expands a log pane.
Position, expanded state, mode and hotkey are remembered between launches.

Launch without a console window:
    pyw -3.12 meddictate_ui.py            # live mode
    pyw -3.12 meddictate_ui.py --batch    # same flags as meddictate.py
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import scrolledtext, ttk

import meddictate as m

HERE = Path(__file__).parent
STATE_FILE = HERE / "widget_state.json"

HOTKEYS = ["f2", "f3", "f4", "f6", "f7", "f8", "f9", "f10", "f12", "pause", "scroll lock"]

DOT = {
    "loading": "#e0a800",
    "idle": "#9a9a9a",
    "recording": "#2ecc40",
    "busy": "#e0a800",
    "error": "#e04040",
}
BG = "#1f1f1f"
BG2 = "#2c2c2c"
BG_ACTIVE = "#3d3d3d"
FG = "#e8e8e8"
DIM = "#9a9a9a"
SMALL = ("Segoe UI", 8)


def key_label(k: str) -> str:
    return k.upper() if k.startswith("f") and k[1:].isdigit() else k.title()


class QueueWriter:
    """stdout replacement: ships printed text to the UI thread."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))

    def flush(self):
        pass


def worker(args, q, settings):
    """Runs meddictate's model load + dictation loop off the UI thread."""

    def on_event(event, detail=""):
        q.put(("event", event, detail))

    m.on_event = on_event
    try:
        on_event("loading")
        m.corrector = m.start_corrector()
        print("Loading MedASR...", end=" ", flush=True)
        t0 = time.time()
        recognizer = m.load_recognizer()
        print(f"done ({time.time() - t0:.1f}s)")
        m.run_dictation(recognizer, settings["hotkey"], not args.no_paste,
                        live=settings["live"], settings=settings)
    except SystemExit as e:
        print(f"\n{e}")
        on_event("error", "see log")
    except Exception:
        print("\n" + traceback.format_exc())
        on_event("error", "crashed, see log")


class Widget:
    def __init__(self, root, args, q, settings):
        self.root = root
        self.q = q
        self.settings = settings
        self.state = "loading"
        self.expanded = False
        self.pulse_on = True
        self.last_other_window = None
        self.saved = {}

        root.title("MedASR Dictation")
        root.configure(bg=BG)
        root.resizable(False, False)
        root.attributes("-topmost", True)
        try:
            root.attributes("-toolwindow", True)     # slim title bar, no taskbar button
        except tk.TclError:
            pass

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Hotkey.TCombobox", fieldbackground=BG2, background=BG2,
                        foreground=FG, arrowcolor=DIM, bordercolor=BG2, lightcolor=BG2,
                        darkcolor=BG2, selectbackground=BG2, selectforeground=FG, padding=1)
        style.map("Hotkey.TCombobox", fieldbackground=[("readonly", BG2), ("disabled", BG)],
                  foreground=[("disabled", DIM)])
        root.option_add("*TCombobox*Listbox.background", BG2)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", BG_ACTIVE)
        root.option_add("*TCombobox*Listbox.font", SMALL)

        head = tk.Frame(root, bg=BG, padx=10, pady=8)
        head.pack(fill="x")

        self.dot = tk.Canvas(head, width=16, height=16, bg=BG, highlightthickness=0)
        self.dot_id = self.dot.create_oval(2, 2, 14, 14, fill=DOT["loading"], outline="")
        self.dot.pack(side="left")

        text = tk.Frame(head, bg=BG)
        text.pack(side="left", padx=(8, 12))
        self.status = tk.Label(text, text="Loading MedASR...", bg=BG, fg=FG,
                               font=("Segoe UI", 10, "bold"), anchor="w", width=32)
        self.status.pack(anchor="w")

        # --- controls row: [Live|Batch]  hotkey [F8 v]  Esc cancels
        ctl = tk.Frame(text, bg=BG)
        ctl.pack(anchor="w", pady=(3, 0))
        seg = tk.Frame(ctl, bg=BG2, padx=1, pady=1)
        seg.pack(side="left")
        self.mode_btns = {}
        for live, label in ((True, "Live"), (False, "Batch")):
            b = tk.Label(seg, text=label, font=SMALL, padx=7, pady=0, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, v=live: self.set_mode(v))
            self.mode_btns[live] = b
        settings.setdefault("at_end", m.PASTE_AT_END)
        m.PASTE_AT_END = settings["at_end"]
        seg2 = tk.Frame(ctl, bg=BG2, padx=1, pady=1)
        seg2.pack(side="left", padx=(8, 0))
        self.paste_btns = {}
        for at_end, label in ((False, "Cursor"), (True, "End")):
            b = tk.Label(seg2, text=label, font=SMALL, padx=7, pady=0, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, v=at_end: self.set_paste_mode(v))
            self.paste_btns[at_end] = b
        tk.Label(ctl, text="hotkey", bg=BG, fg=DIM, font=SMALL).pack(side="left", padx=(10, 4))
        self.hotkey_var = tk.StringVar(value=key_label(settings["hotkey"]))
        self.hotkey_box = ttk.Combobox(ctl, textvariable=self.hotkey_var, width=9, font=SMALL,
                                       values=[key_label(k) for k in HOTKEYS],
                                       state="readonly", style="Hotkey.TCombobox")
        self.hotkey_box.pack(side="left")
        self.hotkey_box.bind("<<ComboboxSelected>>", self.on_hotkey)
        tk.Label(ctl, text="Esc cancels", bg=BG, fg=DIM, font=SMALL).pack(side="left", padx=(10, 0))

        self.toggle = tk.Label(head, text="▾", bg=BG, fg=DIM, font=("Segoe UI", 12), cursor="hand2")
        self.toggle.pack(side="right", anchor="n")
        self.toggle.bind("<Button-1>", lambda e: self.set_expanded(not self.expanded))

        self.body = tk.Frame(root, bg=BG, padx=10)
        self.log = scrolledtext.ScrolledText(
            self.body, height=12, width=58, bg="#121212", fg=FG, insertbackground=FG,
            font=("Consolas", 9), relief="flat", wrap="word", state="disabled",
            borderwidth=0, highlightthickness=0,
        )
        self.log.pack(fill="both", expand=True)
        bar = tk.Frame(self.body, bg=BG)
        bar.pack(fill="x", pady=(6, 0))
        tk.Label(bar, text="Log", bg=BG, fg=DIM, font=SMALL).pack(side="left")
        clear = tk.Label(bar, text="clear", bg=BG, fg=DIM, font=("Segoe UI", 8, "underline"),
                         cursor="hand2")
        clear.pack(side="right")
        clear.bind("<Button-1>", lambda e: self.clear_log())

        self.paint_mode()
        self.restore_state()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(80, self.poll)
        root.after(500, self.pulse)

    # ---- state persistence -------------------------------------------------

    def restore_state(self):
        try:
            st = json.loads(STATE_FILE.read_text())
            self.root.geometry(f"+{int(st['x'])}+{int(st['y'])}")
            self.set_expanded(bool(st.get("expanded", False)))
        except Exception:
            self.root.update_idletasks()
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f"+{sw - self.root.winfo_reqwidth() - 40}+{sh - 160}")
            self.set_expanded(False)

    def save_state(self):
        try:
            STATE_FILE.write_text(json.dumps({
                "x": self.root.winfo_x(), "y": self.root.winfo_y(),
                "expanded": self.expanded,
                "hotkey": self.settings["hotkey"], "live": self.settings["live"],
                "at_end": self.settings["at_end"],
            }))
        except Exception:
            pass

    def quit(self):
        self.save_state()
        if m.corrector is not None:
            m.corrector.stop()
        self.root.destroy()

    # ---- focus handling ----------------------------------------------------
    # Clicking a control makes this widget the foreground window. Hand focus
    # back to the window the user was in so the next hotkey press still locks
    # onto their note rather than onto us.

    def own_hwnd(self):
        try:
            return m.user32.GetAncestor(self.root.winfo_id(), 2)      # GA_ROOT
        except Exception:
            return None

    def track_foreground(self):
        if not m.IS_WINDOWS:
            return
        try:
            fg = m.user32.GetForegroundWindow()
            if fg and fg != self.own_hwnd():
                self.last_other_window = fg
        except Exception:
            pass

    def give_focus_back(self):
        if m.IS_WINDOWS and self.last_other_window:
            try:
                m.restore_window(self.last_other_window)
            except Exception:
                pass

    # ---- controls ----------------------------------------------------------

    def set_mode(self, live: bool):
        if self.state not in ("idle",) or self.settings["live"] == live:
            return
        self.settings["live"] = live
        self.paint_mode()
        self.append_log(f"\nMode -> {'LIVE' if live else 'BATCH'}\n")
        self.save_state()
        self.root.after(50, self.give_focus_back)

    def set_paste_mode(self, at_end: bool):
        if self.state != "idle" or self.settings["at_end"] == at_end:
            return
        self.settings["at_end"] = at_end
        m.PASTE_AT_END = at_end
        self.paint_mode()
        self.append_log(f"\nPaste -> {'append at END of box' if at_end else 'insert at CURSOR'}\n")
        self.save_state()
        self.root.after(50, self.give_focus_back)

    def paint_mode(self):
        for live, b in self.mode_btns.items():
            active = (self.settings["live"] == live)
            b.configure(bg=BG_ACTIVE if active else BG2, fg=FG if active else DIM)
        for at_end, b in self.paste_btns.items():
            active = (self.settings["at_end"] == at_end)
            b.configure(bg=BG_ACTIVE if active else BG2, fg=FG if active else DIM)

    def on_hotkey(self, _event=None):
        label = self.hotkey_var.get()
        key = next((k for k in HOTKEYS if key_label(k) == label), label.lower())
        self.hotkey_box.selection_clear()
        if key != self.settings["hotkey"]:
            self.settings["hotkey"] = key
            self.append_log(f"\nHotkey -> {key_label(key)}\n")
            self.save_state()
            if self.state == "idle":
                self.set_event("idle", "")
        self.root.after(50, self.give_focus_back)

    def set_controls_enabled(self, on: bool):
        self.hotkey_box.configure(state="readonly" if on else "disabled")
        btns = list(self.mode_btns.values()) + list(self.paste_btns.values())
        for b in btns:
            b.configure(cursor="hand2" if on else "arrow")
        if not on:
            for b in btns:
                b.configure(fg=DIM)
        else:
            self.paint_mode()

    # ---- UI updates --------------------------------------------------------

    def set_expanded(self, on):
        self.expanded = on
        if on:
            self.body.pack(fill="both", expand=True, pady=(0, 8))
            self.toggle.configure(text="▴")
        else:
            self.body.pack_forget()
            self.toggle.configure(text="▾")

    def append_log(self, s):
        s = s.replace("\a", "")
        if not s:
            return
        self.log.configure(state="normal")
        self.log.insert("end", s)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def set_event(self, event, detail):
        self.state = event
        self.dot.itemconfigure(self.dot_id, fill=DOT.get(event, DOT["idle"]))
        hk = key_label(self.settings["hotkey"])
        if event == "loading":
            msg = "Loading MedASR..."
        elif event == "idle":
            msg = f"Idle  ·  press {hk} to dictate"
        elif event == "recording":
            where = detail if len(detail) <= 28 else detail[:27] + "…"
            msg = f"Recording  →  {where}"
        elif event == "busy":
            msg = "Transcribing..."
        else:
            msg = f"Error  ·  {detail}" if detail else "Error"
            self.set_expanded(True)
        self.status.configure(text=msg)
        self.set_controls_enabled(event == "idle")

    def poll(self):
        self.track_foreground()
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "log":
                    self.append_log(item[1])
                else:
                    self.set_event(item[1], item[2])
        except queue.Empty:
            pass
        self.root.after(80, self.poll)

    def pulse(self):
        """Blink the dot gently while recording so it's obvious at a glance."""
        if self.state == "recording":
            self.pulse_on = not self.pulse_on
            self.dot.itemconfigure(self.dot_id, fill="#2ecc40" if self.pulse_on else "#1a7f28")
        self.root.after(500, self.pulse)


def initial_settings(args):
    """Saved mode/hotkey win unless overridden explicitly on the command line."""
    settings = {"hotkey": "f8", "live": True, "at_end": m.PASTE_AT_END}
    try:
        st = json.loads(STATE_FILE.read_text())
        if isinstance(st.get("hotkey"), str):
            settings["hotkey"] = st["hotkey"]
        if isinstance(st.get("live"), bool):
            settings["live"] = st["live"]
        if isinstance(st.get("at_end"), bool):
            settings["at_end"] = st["at_end"]
    except Exception:
        pass
    if args.hotkey is not None:
        settings["hotkey"] = args.hotkey.lower()
    if args.batch:
        settings["live"] = False
    return settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true", help="start in batch mode")
    ap.add_argument("--hotkey", default=None, help="start/stop key (default: last used, else f8)")
    ap.add_argument("--no-paste", action="store_true", help="print only, don't paste")
    args = ap.parse_args()

    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on high-DPI screens
    except Exception:
        pass

    q = queue.Queue()
    sys.stdout = QueueWriter(q)
    sys.stderr = QueueWriter(q)
    settings = initial_settings(args)

    root = tk.Tk()
    Widget(root, args, q, settings)
    threading.Thread(target=worker, args=(args, q, settings), daemon=True).start()
    root.mainloop()
    os._exit(0)      # don't wait on the mic/hotkey thread


if __name__ == "__main__":
    main()
