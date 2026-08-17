#!/usr/bin/env python3
"""qde — a browsable terminal UI over the benchmark: experiments, demo, tools.

Arrow keys (or j/k) to browse, Enter to run, q to quit.

Every action runs as a SUBPROCESS rather than in this process. That is deliberate:
if an experiment raises, the UI survives it and shows you the traceback instead of
dying. It also means the commands shown in the preview pane are exactly what you
could have typed by hand -- nothing here is a special path that only works inside
the UI.

Usage:
  python scripts/qde_ui.py            start the UI
  python scripts/qde_ui.py --tui      force the UI even without a detected TTY
  python scripts/qde_ui.py --list     print the whole menu as plain text
  python scripts/qde_ui.py --help     this message
"""
from __future__ import annotations

import os
import subprocess
import sys
import termios
import tty

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
PY = sys.executable

NAVY, BLUE, CORAL, GREEN, MUTED = "#152245", "#008DE3", "#DB4D3A", "#0A9B75", "#6E7A94"


# --------------------------------------------------------------------- items
def build_items():
    from experiments import load_index, producer_for

    items = [("header", "EXPERIMENTS  ·  results chapter", None, None)]
    for i, e in enumerate(load_index(), 1):
        prod = producer_for(e["artifacts"][0]).split("(")[0].strip() if e["artifacts"] else "-"
        detail = Group(
            Text(e["title"], style=f"bold {NAVY}"),
            Text(""),
            Text(f"thesis section   {e['label']}", style=MUTED),
            Text(f"produced by      {prod}", style=MUTED),
            Text(f"artifacts        {len(e['artifacts'])}", style=MUTED),
            Text(""),
            Text("ARTIFACTS", style=f"bold {BLUE}"),
            *[Text(f"  {'ok' if os.path.exists(os.path.join(ROOT, a)) else '--'}  {a}",
                   style=GREEN if os.path.exists(os.path.join(ROOT, a)) else CORAL)
              for a in e["artifacts"]],
            Text(""),
            Text("Enter shows the section, the script, and the result tables.", style=MUTED),
        )
        items.append(("item", f"{i:2d}. {e['title'][:44]}",
                      [PY, "scripts/experiments.py", str(i)], detail))

    items.append(("header", "LIVE DEMO  ·  each step proves one claim", None, None))
    demo = [
        (1, "Environment checks its own arithmetic", "Runs the known-answer guard, then breaks the complex dot product on purpose and shows the guard catching it."),
        (2, "One experiment, end to end", "config -> data -> split -> scale -> features -> ridge -> NRMSE, every stage printed."),
        (3, "Leakage, demonstrated", "Nine system/scaler combinations. Lorenz+standard reports 63% better purely from leaking; Henon+minmax moves 0.01%; two cells get worse."),
        (4, "Leakage, machine-checked", "One query over the run registry: 360 of 360 runs used training-only scaling."),
        (5, "The cache cannot change a result", "Feature matrix with the cache off and on, compared with .tobytes()."),
        (6, "Same configuration twice", "Bit-identical re-execution, printed at full precision."),
    ]
    for n, title, blurb in demo:
        items.append(("item", f" D{n}. {title}",
                      [PY, "scripts/demo.py", "--step", str(n)],
                      Group(Text(f"Demo step {n}", style=f"bold {NAVY}"), Text(""),
                            Text(title, style=f"bold {BLUE}"), Text(""),
                            Text(blurb, style=NAVY))))

    items.append(("header", "TOOLS", None, None))
    tools = [
        ("Trace a number to its source", [PY, "scripts/trace.py", "__ASK__"],
         "Prompts for a number, then resolves it to a registry run (model, seed, dataset hash, split) or to the committed CSV, row and column that holds it."),
        ("Run the test suite", [PY, "-m", "pytest", "tests/", "-q"],
         "131 tests: unit gates, data quality, fault injection, and the known-answer numerical guards."),
        ("List all experiments", [PY, "scripts/experiments.py"],
         "The 19 experiments of the results chapter with artifact counts."),
        ("Full demo, all six steps", [PY, "scripts/demo.py"],
         "About two seconds end to end."),
    ]
    for title, cmd, blurb in tools:
        items.append(("item", f"  *  {title}", cmd,
                      Group(Text(title, style=f"bold {NAVY}"), Text(""), Text(blurb, style=NAVY))))
    return items


# --------------------------------------------------------------------- keys
class Keyboard:
    """Reads single keypresses from the controlling terminal.

    Reads /dev/tty rather than sys.stdin wherever possible, so the UI still works
    when stdin has been redirected or wrapped by a launcher. Falls back to stdin
    only if /dev/tty cannot be opened.
    """

    def __init__(self):
        self.fd = None
        self.own = False
        try:
            self.fd = os.open("/dev/tty", os.O_RDONLY)
            self.own = True
        except OSError:
            if sys.stdin.isatty():
                self.fd = sys.stdin.fileno()

    @property
    def usable(self):
        if self.fd is None:
            return False
        try:
            termios.tcgetattr(self.fd)
            return True
        except Exception:
            return False

    # Raw mode is entered ONCE for the lifetime of the UI, not per keypress.
    # Toggling it around every read loses buffered input: hold an arrow key down
    # and the repeats arrive as a burst, but each restore-to-cooked between reads
    # mangles what is still in the queue, so most of the presses vanish.
    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        self.restore()
        self.close()

    def restore(self):
        if self.fd is not None and getattr(self, "saved", None):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def raw(self):
        if self.fd is not None:
            tty.setraw(self.fd)

    def close(self):
        if self.own and self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def read(self):
        c = os.read(self.fd, 1).decode("utf-8", "replace")
        if c == "\x1b":                           # escape sequence: arrows, page keys
            nxt = os.read(self.fd, 2).decode("utf-8", "replace")
            return {"[A": "up", "[B": "down", "[5": "pgup", "[6": "pgdn"}.get(nxt, "esc")
        return {"\r": "enter", "\n": "enter", "\x03": "quit", "q": "quit",
                "j": "down", "k": "up", " ": "enter"}.get(c, c)


# --------------------------------------------------------------------- render
def render(items, sel, console):
    LW = 46
    def clip(s, n):
        return s if len(s) <= n else s[:n - 1] + "…"

    left = Table.grid(padding=(0, 1))
    left.add_column(width=LW, overflow="crop", no_wrap=True)
    height = max(8, console.size.height - 8)
    idx = [i for i, it in enumerate(items) if it[0] == "item"]
    pos = idx.index(sel)
    lo = max(0, min(pos - height // 2, len(items) - height))
    for i, (kind, label, _cmd, _d) in enumerate(items[lo:lo + height], start=lo):
        # clip rather than wrap: a wrapped row shifts every row under it and the
        # selection highlight stops lining up with what it points at.
        if kind == "header":
            left.add_row(Text(clip(label, LW), style=f"bold {CORAL}", no_wrap=True))
        elif i == sel:
            left.add_row(Text(clip(f"> {label}", LW).ljust(LW),
                              style=f"bold white on {BLUE}", no_wrap=True))
        else:
            left.add_row(Text(clip(f"  {label}", LW), style=NAVY, no_wrap=True))

    detail = items[sel][3] or Text("")
    cmd = items[sel][2]
    cmdline = " ".join(["python" if c == PY else c for c in cmd]) if cmd else ""
    right = Group(detail, Text(""),
                  Text("COMMAND", style=f"bold {MUTED}"),
                  Text(f"  {cmdline}", style=GREEN))

    body = Table.grid(padding=(0, 2), expand=True)
    body.add_column(width=48)
    body.add_column(ratio=1)
    body.add_row(Panel(left, title="browse", border_style=MUTED, padding=(0, 1)),
                 Panel(right, title="detail", border_style=MUTED, padding=(1, 2)))

    console.print(Panel(
        Align.center(Text("QDE  ·  quantum vs classical reservoir computing on chaotic time series",
                          style=f"bold {NAVY}")),
        border_style=BLUE, padding=(0, 1)))
    console.print(body)
    console.print(Align.center(Text(
        "  up/down move   ·   enter run   ·   q quit  ", style=MUTED)))


def run(cmd, console, kb):
    # Hand the terminal back to cooked mode for the child: it needs line editing
    # for prompts, and a subprocess inheriting raw mode behaves strangely.
    kb.restore()
    try:
        if "__ASK__" in cmd:
            console.print()
            try:
                val = console.input(f"[{BLUE}]number to trace:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not val:
                return
            cmd = [c for c in cmd if c != "__ASK__"] + [val]
        shown = " ".join("python" if c == PY else c for c in cmd)
        console.print()
        console.print(Panel(Text(f"$ {shown}", style=GREEN), border_style=GREEN, padding=(0, 1)))
        console.print()
        try:
            # inherits stdout, so output streams live exactly as in a shell
            subprocess.run(cmd, cwd=ROOT)
        except Exception as e:                                # never kill the UI
            console.print(Text(f"command failed: {e}", style=CORAL))
        console.print()
        console.print(Text("  press any key to go back", style=MUTED))
    finally:
        kb.raw()
    kb.read()


def print_list(items):
    # plain print, not rich: this path exists for piping and for terminals that
    # are not a TTY, where rich's wrapping splits the commands across lines.
    for kind, label, cmd, _ in items:
        if kind == "header":
            print(f"\n{label}")
        else:
            shown = " ".join("python" if c == PY else c for c in cmd)
            print(f"  {label:<50s} {shown}")


def main():
    items = build_items()
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__.strip())
        return

    want_list = "--list" in sys.argv
    force_tui = "--tui" in sys.argv

    kb = Keyboard()
    if want_list or not (force_tui or kb.usable):
        if not want_list:
            # Never fall back silently. The first version did, and the only
            # symptom was "it prints a list and exits" with no reason given.
            print("qde_ui: no controlling terminal available, so the interactive UI")
            print("        cannot start. Showing the menu as plain text instead.")
            print("        Run it directly in a terminal, or force it with --tui.")
            print(f"        (stdin tty={sys.stdin.isatty()}, /dev/tty="
                  f"{'ok' if kb.fd is not None else 'unavailable'})")
        print_list(items)
        kb.close()
        return

    # force_terminal: the UI only starts when a real terminal is present, so tell
    # rich to render colour even if it misdetects a wrapped stdout.
    console = Console(force_terminal=True)
    sel = next(i for i, it in enumerate(items) if it[0] == "item")
    idx = [i for i, it in enumerate(items) if it[0] == "item"]
    try:
        with kb:
            while True:
                console.clear()
                render(items, sel, console)
                k = kb.read()
                if k == "quit":
                    break
                if k in ("up", "down", "pgup", "pgdn"):
                    step = {"up": -1, "down": 1, "pgup": -8, "pgdn": 8}[k]
                    p = min(max(idx.index(sel) + step, 0), len(idx) - 1)
                    sel = idx[p]
                elif k == "enter":
                    console.clear()
                    run(items[sel][2], console, kb)
    except KeyboardInterrupt:
        pass
    except Exception:
        # show the failure rather than vanishing into a restored terminal
        console.print()
        import traceback
        traceback.print_exc()
        console.print(Text("qde_ui: the UI hit an error and stopped; "
                           "the commands above still work directly.", style=CORAL))
    console.print(Text("bye", style=MUTED))


if __name__ == "__main__":
    main()
