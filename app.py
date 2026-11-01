import json
import queue
import threading
import tkinter as tk
import urllib.request
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

WHISPER_MODEL = "base.en"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.03          # must be 0.01, 0.02 or 0.03 for webrtcvad
VAD_MODE = 2                  # 0 = least aggressive filtering, 3 = most aggressive
SILENCE_END_SECONDS = 0.8
MAX_UTTERANCE_SECONDS = 15
BEAM_SIZE = 5
MIC_DEVICE = None             # None = system default input device; set to a device index to force one

BG = "#0b0b12"
PANEL = "#15151f"
PANEL_BORDER = "#242433"
ACCENT = "#a78bfa"
ACCENT_2 = "#34d399"
DANGER = "#f87171"
TEXT = "#e6e6f0"
SUBTEXT = "#8a8aa3"
MUTED_BTN = "#1f1f2e"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Local Answer Assistant")
        root.geometry("760x660")
        root.minsize(560, 480)
        root.configure(bg=BG)

        self.ui_q: queue.Queue = queue.Queue()
        self.audio_q: queue.Queue = queue.Queue()
        self.recording = threading.Event()
        self.stream = None
        self.model = None
        self.stt_thread = None
        self._pulse_phase = 0

        self._build_fonts()
        self._build_style()
        self._build_ui()

        self.root.after(50, self.drain_ui_queue)
        self.root.after(80, self._pulse_dot)

    def _build_fonts(self):
        families = set(tkfont.families())
        display = next((f for f in ("Space Grotesk", "Poppins", "Segoe UI Variable",
                                     "Segoe UI", "SF Pro Display", "Helvetica Neue")
                         if f in families), "TkDefaultFont")
        body = next((f for f in ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue",
                                  "Helvetica") if f in families), "TkDefaultFont")
        mono = next((f for f in ("JetBrains Mono", "Cascadia Code", "Consolas", "Menlo",
                                  "Courier New") if f in families), "TkFixedFont")
        self.f_title = tkfont.Font(family=display, size=18, weight="bold")
        self.f_label = tkfont.Font(family=body, size=10, weight="bold")
        self.f_status = tkfont.Font(family=body, size=10)
        self.f_button = tkfont.Font(family=body, size=11, weight="bold")
        self.f_text = tkfont.Font(family=mono, size=11)

    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Dark.TLabel", background=BG, foreground=TEXT)
        style.configure("Sub.TLabel", background=BG, foreground=SUBTEXT, font=self.f_status)
        style.configure("Section.TLabel", background=BG, foreground=SUBTEXT, font=self.f_label)

        def pill(name, bg, fg, active_bg):
            style.configure(name, background=bg, foreground=fg, font=self.f_button,
                             borderwidth=0, focuscolor=bg, padding=(18, 10))
            style.map(name,
                      background=[("disabled", "#111118"), ("active", active_bg)],
                      foreground=[("disabled", "#45455a")])

        pill("Start.TButton", ACCENT_2, "#06241a", "#5eead4")
        pill("Stop.TButton", DANGER, "#2a0a0a", "#fca5a5")
        pill("Suggest.TButton", ACCENT, "#1c1033", "#c4b5fd")

    def _card(self, parent):
        outer = tk.Frame(parent, bg=PANEL_BORDER)
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return outer, inner

    def _build_ui(self):
        root = self.root

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=22, pady=(20, 10))

        title_row = tk.Frame(header, bg=BG)
        title_row.pack(fill="x")
        tk.Label(title_row, text="⚡ answer assistant", font=self.f_title,
                 bg=BG, fg=TEXT).pack(side="left")

        status_row = tk.Frame(title_row, bg=BG)
        status_row.pack(side="right")
        self.dot = tk.Canvas(status_row, width=12, height=12, bg=BG, highlightthickness=0)
        self._dot_id = self.dot.create_oval(1, 1, 11, 11, fill=SUBTEXT, outline="")
        self.dot.pack(side="left", padx=(0, 8))
        self.status = tk.Label(status_row, text="idle", font=self.f_status, bg=BG, fg=SUBTEXT)
        self.status.pack(side="left")

        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=22, pady=(6, 16))
        self.start_btn = ttk.Button(bar, text="▶  Start", style="Start.TButton", command=self.start)
        self.stop_btn = ttk.Button(bar, text="■  Stop", style="Stop.TButton", command=self.stop,
                                    state="disabled")
        self.suggest_btn = ttk.Button(bar, text="✦  Suggest Answer", style="Suggest.TButton",
                                       command=self.suggest)
        self.start_btn.pack(side="left")
        self.stop_btn.pack(side="left", padx=(10, 0))
        self.suggest_btn.pack(side="left", padx=(10, 0))

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 20))
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)

        t_wrap = tk.Frame(body, bg=BG)
        t_wrap.grid(row=0, column=0, sticky="nsew", pady=(0, 14))
        tk.Label(t_wrap, text="TRANSCRIPT", font=self.f_label, bg=BG, fg=ACCENT_2).pack(
            anchor="w", pady=(0, 6))
        t_outer, t_inner = self._card(t_wrap)
        t_outer.pack(fill="both", expand=True)
        self.transcript = tk.Text(t_inner, wrap="word", bg=PANEL, fg=TEXT,
                                   insertbackground=TEXT, relief="flat", bd=0,
                                   font=self.f_text, padx=14, pady=12)
        t_scroll = ttk.Scrollbar(t_inner, command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=t_scroll.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        t_scroll.pack(side="right", fill="y")

        a_wrap = tk.Frame(body, bg=BG)
        a_wrap.grid(row=1, column=0, sticky="nsew")
        tk.Label(a_wrap, text="SUGGESTED ANSWER", font=self.f_label, bg=BG, fg=ACCENT).pack(
            anchor="w", pady=(0, 6))
        a_outer, a_inner = self._card(a_wrap)
        a_outer.pack(fill="both", expand=True)
        self.answer = tk.Text(a_inner, wrap="word", bg=PANEL, fg=TEXT,
                               insertbackground=TEXT, relief="flat", bd=0,
                               font=self.f_text, padx=14, pady=12)
        a_scroll = ttk.Scrollbar(a_inner, command=self.answer.yview)
        self.answer.configure(yscrollcommand=a_scroll.set)
        self.answer.pack(side="left", fill="both", expand=True)
        a_scroll.pack(side="right", fill="y")

    def _set_status(self, text: str, color: str):
        self.status.config(text=text, fg=color)
        self.dot.itemconfig(self._dot_id, fill=color)

    def _pulse_dot(self):
        if self.recording.is_set() or self._busy:
            self._pulse_phase = (self._pulse_phase + 1) % 20
            r = 1 + abs(10 - self._pulse_phase) * 0.15
            cx, cy = 6, 6
            self.dot.coords(self._dot_id, cx - 5 - r, cy - 5 - r, cx + 5 + r, cy + 5 + r)
        self.root.after(80, self._pulse_dot)

    @property
    def _busy(self):
        return getattr(self, "_llm_busy", False)

    def drain_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_q.get_nowait()
                if kind == "status":
                    self._set_status(payload["text"], payload.get("color", SUBTEXT))
                elif kind == "transcript":
                    self.transcript.insert("end", payload + " ")
                    self.transcript.see("end")
                elif kind == "answer":
                    self.answer.insert("end", payload)
                    self.answer.see("end")
                elif kind == "suggest_done":
                    self._llm_busy = False
                    self.suggest_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(50, self.drain_ui_queue)

    def start(self):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.recording.set()
        self.stt_thread = threading.Thread(target=self.stt_worker, daemon=True)
        self.stt_thread.start()

    def stop(self):
        self.recording.clear()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.ui_q.put(("status", {"text": "stopped", "color": SUBTEXT}))

    def stt_worker(self):
        if self.model is None:
            self.ui_q.put(("status", {"text": "loading speech model…", "color": ACCENT}))
            self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

        vad = webrtcvad.Vad(VAD_MODE)

        def callback(indata, frames, time_info, status):
            self.audio_q.put(indata[:, 0].copy())

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=MIC_DEVICE,
                blocksize=int(SAMPLE_RATE * BLOCK_SECONDS), callback=callback)
            self.stream.start()
        except Exception as e:
            devices = "\n".join(f"  [{i}] {d['name']}" for i, d in enumerate(sd.query_devices())
                                 if d["max_input_channels"] > 0)
            self.ui_q.put(("status", {"text": f"mic error: {e}", "color": DANGER}))
            self.ui_q.put(("transcript", f"\n[Could not open microphone. Available input "
                                          f"devices:\n{devices}\nSet MIC_DEVICE in app.py to "
                                          f"the index you want.]\n"))
            self.root.after(0, self.stop)
            return

        self.ui_q.put(("status", {"text": "listening…", "color": ACCENT_2}))
        buf, silent_blocks, speaking = [], 0, False
        end_blocks = int(SILENCE_END_SECONDS / BLOCK_SECONDS)
        max_blocks = int(MAX_UTTERANCE_SECONDS / BLOCK_SECONDS)

        while self.recording.is_set():
            try:
                block = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            pcm16 = np.clip(block * 32767, -32768, 32767).astype(np.int16).tobytes()
            is_speech = vad.is_speech(pcm16, SAMPLE_RATE)
            if is_speech:
                speaking, silent_blocks = True, 0
                buf.append(block)
            elif speaking:
                silent_blocks += 1
                buf.append(block)
            if speaking and (silent_blocks >= end_blocks or len(buf) >= max_blocks):
                self.transcribe(np.concatenate(buf))
                buf, silent_blocks, speaking = [], 0, False

        if buf:
            self.transcribe(np.concatenate(buf))

    def transcribe(self, audio: np.ndarray):
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True, beam_size=BEAM_SIZE)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            self.ui_q.put(("transcript", text))

    def suggest(self):
        question = self.transcript.get("1.0", "end").strip()
        if not question:
            self.ui_q.put(("status", {"text": "nothing transcribed yet", "color": DANGER}))
            return
        question = question[-2000:]
        self.answer.delete("1.0", "end")
        self._llm_busy = True
        self.suggest_btn.config(state="disabled")
        threading.Thread(target=self.llm_worker, args=(question,), daemon=True).start()

    def llm_worker(self, question: str):
        self.ui_q.put(("status", {"text": "generating answer…", "color": ACCENT}))
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content":
                    "You help the user answer questions they just heard. The text is a "
                    "live speech transcript and may contain errors. Answer the most recent "
                    "question clearly and concisely, as the user would say it."},
                {"role": "user", "content": question},
            ],
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        self.ui_q.put(("answer", token))
                    if chunk.get("done"):
                        break
            self.ui_q.put(("status", {"text": "done", "color": ACCENT_2}))
        except Exception as e:
            self.ui_q.put(("answer", f"[LLM error: {e}\nIs Ollama running and "
                                     f"'{OLLAMA_MODEL}' pulled?]"))
            self.ui_q.put(("status", {"text": "LLM error", "color": DANGER}))
        finally:
            self.ui_q.put(("suggest_done", None))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()