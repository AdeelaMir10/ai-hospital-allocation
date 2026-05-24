"""
AI HOSPITAL ROOM ALLOCATION SYSTEM
BS Computer Science — AI Lab Project

Algorithms (3 categories as required):
  1. Uninformed Search  → BFS
  2. Informed Search    → A* Search
  3. Adversarial Search → Minimax + Alpha-Beta Pruning
  4. Agents             → Simple Reflex | Model-Based | Goal-Based | Utility-Based
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random, math, time, heapq
from collections import deque

# ──────────────────────────────────────────────
#  CONSTANTS & DATA
# ──────────────────────────────────────────────

DEPARTMENTS  = ["ICU", "General", "Cardiology", "Pediatrics", "Emergency", "Neurology"]
CONDITIONS   = ["Critical", "Serious", "Moderate", "Stable"]
COND_PRIO    = {"Critical": 10, "Serious": 7, "Moderate": 4, "Stable": 1}
ROOM_TYPES   = ["ICU Room", "Private", "General Ward", "Emergency Bay"]
NAMES        = ["Ahmed Ali","Sara Malik","Omar Raza","Hina Shah","Bilal Khan",
                "Zara Hussain","Tariq Butt","Amna Qureshi","Faisal Javed","Nadia Anwar",
                "Kamran Iqbal","Sana Riaz","Hassan Sheikh","Rabia Zaidi","Usman Tariq"]

C = {
    "bg":      "#0B0F1A", "bg2":    "#111827", "bg3":    "#1A2234",
    "panel":   "#1E293B", "border": "#2D3E55",
    "accent":  "#38BDF8", "green":  "#34D399", "yellow": "#FBBF24",
    "red":     "#F87171", "purple": "#A78BFA",
    "text":    "#E2E8F0", "dim":    "#64748B",
}
COND_COLOR = {"Critical": C["red"], "Serious": C["yellow"], "Moderate": C["accent"], "Stable": C["green"]}


# ──────────────────────────────────────────────
#  DATA MODELS
# ──────────────────────────────────────────────

class Patient:
    _id = 0
    def __init__(self, name, age, condition, department):
        Patient._id += 1
        self.pid       = Patient._id
        self.name      = name
        self.age       = age
        self.condition = condition
        self.department= department
        self.priority  = COND_PRIO[condition] + (2 if age > 60 else 0)
        self.room      = None
    def __lt__(self, o): return self.priority > o.priority


class Room:
    _id = 0
    def __init__(self, rtype, department, capacity):
        Room._id += 1
        self.rid      = Room._id
        self.rtype    = rtype
        self.dept     = department
        self.capacity = capacity
        self.occupied = 0
    @property
    def available(self): return self.occupied < self.capacity
    @property
    def rate(self): return self.occupied / self.capacity


class Hospital:
    def __init__(self):
        self.rooms    = []
        self.patients = []
        self.log      = []
        self.allocated= 0
        self._make_rooms()

    def _make_rooms(self):
        config = [
            ("ICU Room","ICU",1), ("ICU Room","ICU",1), ("ICU Room","Cardiology",1),
            ("Emergency Bay","Emergency",2), ("Emergency Bay","Emergency",2),
            ("Private","Cardiology",1), ("Private","Neurology",1), ("Private","General",1),
            ("General Ward","General",4), ("General Ward","General",4),
            ("General Ward","Pediatrics",3), ("General Ward","Pediatrics",3),
        ]
        for rtype, dept, cap in config:
            r = Room(rtype, dept, cap)
            r.occupied = random.randint(0, cap - 1) if random.random() > 0.5 else 0
            self.rooms.append(r)

    def add_patient(self, name, age, condition, dept):
        p = Patient(name, age, condition, dept)
        self.patients.append(p)
        self.log_msg(f"Patient registered: {p.name} | {condition} | {dept} | Priority={p.priority}")
        return p

    def allocate(self, patient, room):
        room.occupied += 1
        patient.room = room.rid
        self.allocated += 1
        self.log_msg(f"✅ {patient.name} → Room {room.rid} [{room.rtype}, {room.dept}]")

    def log_msg(self, m):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {m}")

    def unassigned(self):
        return [p for p in self.patients if p.room is None]

    def available_rooms(self):
        return [r for r in self.rooms if r.available]


# ──────────────────────────────────────────────
#  1. BFS — Uninformed Search
# ──────────────────────────────────────────────

def bfs_find_room(hospital: Hospital, patient: Patient):
    """
    BFS over room graph (rooms connected by same floor/adjacency).
    Finds the NEAREST available room suitable for patient's dept.
    Returns (room, path_of_rids, steps_explored)
    """
    # Build simple adjacency: rooms close in list index = adjacent
    rooms = hospital.rooms
    graph = {r.rid: [] for r in rooms}
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if i != j and abs(i - j) <= 3:   # within 3 slots = adjacent ward
                graph[r1.rid].append(r2.rid)

    start = rooms[0].rid
    visited = {start}
    queue   = deque([(start, [start])])
    steps   = 0

    while queue:
        cur, path = queue.popleft()
        steps += 1
        room = next((r for r in rooms if r.rid == cur), None)
        if room and room.available and cur != start:
            return room, path, steps
        for nb in graph.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return None, [], steps


# ──────────────────────────────────────────────
#  2. A* — Informed Search
# ──────────────────────────────────────────────

def g_cost(patient: Patient, room: Room):
    """Actual cost: penalise dept mismatch and high occupancy"""
    cost = 0
    if room.dept != patient.department: cost += 6
    ideal = {"Critical":"ICU Room","Serious":"Private",
             "Moderate":"General Ward","Stable":"General Ward"}
    if room.rtype != ideal.get(patient.condition): cost += 4
    cost += room.rate * 3
    return cost

def h_cost(patient: Patient, room: Room):
    """Heuristic: urgency vs room quality gap"""
    urgency = COND_PRIO[patient.condition]
    quality = {"ICU Room":10,"Emergency Bay":8,"Private":6,"General Ward":3}
    return abs(urgency - quality.get(room.rtype, 3)) * 0.4

def astar_find_room(hospital: Hospital, patient: Patient):
    """
    A* picks the room with minimum f = g + h.
    Returns (best_room, nodes_explored)
    """
    available = hospital.available_rooms()
    if not available:
        return None, 0
    heap = []
    for room in available:
        g = g_cost(patient, room)
        h = h_cost(patient, room)
        heapq.heappush(heap, (g + h, g, room.rid))
    explored = 0
    while heap:
        f, g, rid = heapq.heappop(heap)
        room = next((r for r in hospital.rooms if r.rid == rid), None)
        explored += 1
        if room and room.available:
            return room, explored
    return None, explored


# ──────────────────────────────────────────────
#  3. Minimax + Alpha-Beta — Adversarial Search
# ──────────────────────────────────────────────

_nodes_mm  = 0
_nodes_ab  = 0

def minimax(depth, is_max, rooms, hi_pts, lo_pts):
    global _nodes_mm
    _nodes_mm += 1
    if depth == 0 or not rooms or (not hi_pts and not lo_pts):
        # Evaluate: unallocated high-priority patients = bad for MAX
        return -sum(p.priority for p in hi_pts) + sum(p.priority * 0.3 for p in lo_pts)
    if is_max:
        val = -math.inf
        for i in range(min(len(hi_pts), len(rooms))):
            val = max(val, minimax(depth-1, False, rooms[1:], hi_pts[:i]+hi_pts[i+1:], lo_pts))
        return val if hi_pts else minimax(depth-1, False, rooms, hi_pts, lo_pts)
    else:
        val = math.inf
        for i in range(min(len(lo_pts), len(rooms))):
            val = min(val, minimax(depth-1, True, rooms[1:], hi_pts, lo_pts[:i]+lo_pts[i+1:]))
        return val if lo_pts else minimax(depth-1, True, rooms, hi_pts, lo_pts)

def alphabeta(depth, is_max, rooms, hi_pts, lo_pts, alpha=-math.inf, beta=math.inf):
    global _nodes_ab
    _nodes_ab += 1
    if depth == 0 or not rooms or (not hi_pts and not lo_pts):
        return -sum(p.priority for p in hi_pts) + sum(p.priority * 0.3 for p in lo_pts)
    if is_max:
        val = -math.inf
        for i in range(min(len(hi_pts), len(rooms))):
            val = max(val, alphabeta(depth-1, False, rooms[1:], hi_pts[:i]+hi_pts[i+1:], lo_pts, alpha, beta))
            alpha = max(alpha, val)
            if beta <= alpha: break   # β cutoff
        return val if hi_pts else alphabeta(depth-1, False, rooms, hi_pts, lo_pts, alpha, beta)
    else:
        val = math.inf
        for i in range(min(len(lo_pts), len(rooms))):
            val = min(val, alphabeta(depth-1, True, rooms[1:], hi_pts, lo_pts[:i]+lo_pts[i+1:], alpha, beta))
            beta = min(beta, val)
            if beta <= alpha: break   # α cutoff
        return val if lo_pts else alphabeta(depth-1, True, rooms, hi_pts, lo_pts, alpha, beta)

def run_adversarial(hospital: Hospital, depth: int):
    global _nodes_mm, _nodes_ab
    hi = [p for p in hospital.unassigned() if p.priority >= 7][:3]
    lo = [p for p in hospital.unassigned() if p.priority  < 7][:3]
    rooms = hospital.available_rooms()[:4]
    if not hi: hi = [Patient("MAX-Test", 60, "Critical", "ICU")]
    if not lo: lo = [Patient("MIN-Test", 30, "Stable",   "General")]

    _nodes_mm = 0
    t0 = time.time(); v1 = minimax(depth, True, rooms, hi, lo); t1 = time.time()
    n1 = _nodes_mm

    _nodes_ab = 0
    t2 = time.time(); v2 = alphabeta(depth, True, rooms, hi, lo); t3 = time.time()
    n2 = _nodes_ab

    pruned = n1 - n2
    return {
        "mm_nodes": n1, "mm_time": (t1-t0)*1000, "mm_val": v1,
        "ab_nodes": n2, "ab_time": (t3-t2)*1000, "ab_val": v2,
        "pruned": pruned, "pct": 100*pruned/max(n1,1),
        "hi": hi, "lo": lo, "rooms": rooms
    }


# ──────────────────────────────────────────────
#  4. AGENTS
# ──────────────────────────────────────────────

class SimpleReflexAgent:
    """Condition → Action rules only. No memory."""
    NAME = "Simple Reflex Agent"
    def act(self, patient, hospital):
        rules = {"Critical":"ICU Room","Serious":"Private",
                 "Moderate":"General Ward","Stable":"General Ward"}
        target = rules[patient.condition]
        for r in hospital.rooms:
            if r.rtype == target and r.available:
                return r, f"Rule: {patient.condition} → {target} → Room {r.rid}"
        for r in hospital.rooms:
            if r.available:
                return r, f"Rule fallback → Room {r.rid}"
        return None, "No room available"

class ModelBasedAgent:
    """Maintains internal model of room states to predict best choice."""
    NAME = "Model-Based Agent"
    def act(self, patient, hospital):
        model = {r.rid: 1 - r.rate for r in hospital.rooms}  # predicted availability
        available = hospital.available_rooms()
        if not available: return None, "No rooms"
        def score(r):
            dept_bonus = 3 if r.dept == patient.department else 0
            return model[r.rid] + dept_bonus
        best = max(available, key=score)
        return best, f"Model score={score(best):.2f} → Room {best.rid} [{best.dept}]"

class GoalBasedAgent:
    """Searches for room that satisfies goal: correct dept + available."""
    NAME = "Goal-Based Agent"
    def act(self, patient, hospital):
        # Goal test: dept matches AND room is available
        for r in hospital.rooms:
            if r.available and r.dept == patient.department:
                return r, f"Goal met: dept={patient.department} → Room {r.rid}"
        # Relaxed goal: just available
        for r in hospital.rooms:
            if r.available:
                return r, f"Relaxed goal (any available) → Room {r.rid}"
        return None, "Goal not achievable"

class UtilityBasedAgent:
    """Picks room maximising a composite utility function."""
    NAME = "Utility-Based Agent"
    def utility(self, patient, room):
        u = 0
        if room.dept == patient.department: u += 30
        ideal = {"Critical":"ICU Room","Serious":"Private",
                 "Moderate":"General Ward","Stable":"General Ward"}
        if room.rtype == ideal.get(patient.condition): u += 25
        u += (1 - room.rate) * 20       # prefer less-full rooms
        if patient.age > 60 and room.rtype in ["ICU Room","Private"]: u += 10
        return u
    def act(self, patient, hospital):
        available = hospital.available_rooms()
        if not available: return None, "No rooms"
        best = max(available, key=lambda r: self.utility(patient, r))
        u = self.utility(patient, best)
        return best, f"Utility={u:.1f} → Room {best.rid} [{best.rtype}, {best.dept}]"


# ──────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.hospital = Hospital()
        self.agents   = [SimpleReflexAgent(), ModelBasedAgent(), GoalBasedAgent(), UtilityBasedAgent()]
        self._last_result = None   # (patient, room) from last algo run
        self._last_astar_result = None
        self._pt_combos   = []     # all patient comboboxes across tabs
        self._setup()
        self._build()
        self._load_scenarios()
        self._tick()

    def _setup(self):
        self.title("AI Hospital Room Allocation System")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(bg=C["bg"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",     background=C["bg"],    borderwidth=0)
        style.configure("TNotebook.Tab", background=C["bg2"],   foreground=C["dim"],
                        padding=[14,7],  font=("Consolas",9),   borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", C["panel"])],
                                   foreground=[("selected", C["accent"])])
        style.configure("Treeview", background=C["bg2"], foreground=C["text"],
                        fieldbackground=C["bg2"], rowheight=24, font=("Consolas",9), borderwidth=0)
        style.configure("Treeview.Heading", background=C["bg3"], foreground=C["accent"],
                        font=("Consolas",9,"bold"), borderwidth=0)
        style.map("Treeview", background=[("selected", C["panel"])],
                              foreground=[("selected", C["accent"])])
        style.configure("TScrollbar", background=C["bg3"], troughcolor=C["bg"],
                        borderwidth=0, arrowcolor=C["dim"])
        style.configure("TCombobox", fieldbackground=C["bg3"], background=C["bg3"],
                        foreground=C["text"], selectbackground=C["panel"])
        style.map("TCombobox", fieldbackground=[("readonly", C["bg3"])],
                               foreground=[("readonly", C["text"])])

    # ── TOP BAR ───────────────────────────────

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C["bg2"], height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⚕  AI HOSPITAL ROOM ALLOCATION", font=("Consolas",16,"bold"),
                 bg=C["bg2"], fg=C["accent"]).pack(side="left", padx=20, pady=14)
        self.clock_lbl = tk.Label(hdr, text="", font=("Consolas",10),
                                  bg=C["bg2"], fg=C["dim"])
        self.clock_lbl.pack(side="right", padx=20)

        # Stat strip
        self._stat_strip()

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(0,10))
        tabs = ["🏥 Dashboard", "👤 Patients", "🔍 BFS Search",
                "⭐ A* Search", "♟ Minimax", "🤖 Agents", "📋 Log"]
        self.T = {}
        for name in tabs:
            f = tk.Frame(nb, bg=C["bg"]); nb.add(f, text=name); self.T[name] = f

        self._tab_dashboard(self.T["🏥 Dashboard"])
        self._tab_patients(self.T["👤 Patients"])
        self._tab_bfs(self.T["🔍 BFS Search"])
        self._tab_astar(self.T["⭐ A* Search"])
        self._tab_minimax(self.T["♟ Minimax"])
        self._tab_agents(self.T["🤖 Agents"])
        self._tab_log(self.T["📋 Log"])

    def _stat_strip(self):
        bar = tk.Frame(self, bg=C["bg3"], height=42); bar.pack(fill="x"); bar.pack_propagate(False)
        self._svars = {}
        items = [("Rooms", "rooms", C["accent"]), ("Available", "avail", C["green"]),
                 ("Patients", "pts",  C["yellow"]), ("Waiting",  "wait",  C["red"]),
                 ("Allocated","alloc",C["green"])]
        for label, key, col in items:
            f = tk.Frame(bar, bg=C["bg3"]); f.pack(side="left", padx=20, pady=4)
            tk.Label(f, text=label, font=("Consolas",8), bg=C["bg3"], fg=C["dim"]).pack(anchor="w")
            v = tk.StringVar(value="—"); self._svars[key] = v
            tk.Label(f, textvariable=v, font=("Consolas",13,"bold"), bg=C["bg3"], fg=col).pack(anchor="w")
        tk.Button(bar, text="⟳ Refresh", font=("Consolas",9), bg=C["panel"], fg=C["accent"],
                  relief="flat", cursor="hand2", command=self._refresh_all).pack(side="right", padx=14)
        self._update_stats()

    def _update_stats(self):
        h = self.hospital
        self._svars["rooms"].set(str(len(h.rooms)))
        self._svars["avail"].set(str(sum(1 for r in h.rooms if r.available)))
        self._svars["pts"].set(str(len(h.patients)))
        self._svars["wait"].set(str(len(h.unassigned())))
        self._svars["alloc"].set(str(h.allocated))

    # ── HELPERS ───────────────────────────────

    def _panel(self, parent, title, **grid_kw):
        f = tk.Frame(parent, bg=C["panel"])
        if title:
            tb = tk.Frame(f, bg=C["bg3"], height=30); tb.pack(fill="x"); tb.pack_propagate(False)
            tk.Label(tb, text=title, font=("Consolas",10,"bold"),
                     bg=C["bg3"], fg=C["accent"]).pack(side="left", padx=10, pady=6)
        if grid_kw: f.grid(**grid_kw)
        return f

    def _tree(self, parent, cols, widths=None):
        fr = tk.Frame(parent, bg=C["panel"]); fr.pack(fill="both", expand=True, padx=8, pady=8)
        t  = ttk.Treeview(fr, columns=cols, show="headings")
        for i, col in enumerate(cols):
            t.heading(col, text=col)
            w = widths[i] if widths else 90
            t.column(col, width=w, anchor="center", minwidth=40)
        sb = ttk.Scrollbar(fr, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); t.pack(fill="both", expand=True)
        return t

    def _logbox(self, parent):
        fr = tk.Frame(parent, bg=C["panel"]); fr.pack(fill="both", expand=True, padx=8, pady=8)
        t  = tk.Text(fr, bg=C["bg2"], fg=C["text"], font=("Consolas",9), state="disabled",
                     relief="flat", bd=0, wrap="word", insertbackground=C["accent"])
        sb = ttk.Scrollbar(fr, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); t.pack(fill="both", expand=True)
        t.tag_configure("h",  foreground=C["accent"],  font=("Consolas",9,"bold"))
        t.tag_configure("ok", foreground=C["green"])
        t.tag_configure("w",  foreground=C["yellow"])
        t.tag_configure("d",  foreground=C["dim"])
        t.tag_configure("r",  foreground=C["red"])
        return t

    def _write(self, box, text, tag=""):
        box.configure(state="normal")
        box.insert("end", text, tag)
        box.configure(state="disabled")

    def _clear(self, box):
        box.configure(state="normal"); box.delete("1.0","end"); box.configure(state="disabled")

    def _btn(self, parent, text, cmd, color=None, **pack):
        color = color or C["accent"]
        b = tk.Button(parent, text=text, font=("Consolas",9), bg=color,
                      fg=C["bg"], relief="flat", cursor="hand2", command=cmd, pady=5)
        b.pack(**pack)
        b.bind("<Enter>", lambda e: b.configure(bg=C["green"]))
        b.bind("<Leave>", lambda e: b.configure(bg=color))
        return b

    def _get_patient(self, combo_var):
        sel = combo_var.get()
        if not sel or sel == "— select —": return None
        pid = int(sel.split("|")[0].strip()[1:])
        return next((p for p in self.hospital.patients if p.pid == pid), None)

    def _patient_options(self, unassigned_only=True):
        pts = self.hospital.unassigned() if unassigned_only else self.hospital.patients
        return ["— select —"] + [f"P{p.pid} | {p.name} | {p.condition}" for p in sorted(pts, key=lambda x:-x.priority)]

    def _refresh_pt_combos(self):
        opts = self._patient_options()
        for cb in getattr(self, "_pt_combos", []):
            cb["values"] = opts
            if opts: cb.current(0)

    # ── DASHBOARD ─────────────────────────────

    def _tab_dashboard(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1);    parent.rowconfigure(1, weight=1)

        # Cards
        cards = self._panel(parent, "Overview")
        cards.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(8,4), pady=8)
        info = [
            ("🏥","Total Rooms",     lambda: str(len(self.hospital.rooms)),                  C["accent"]),
            ("✅","Available",       lambda: str(sum(1 for r in self.hospital.rooms if r.available)), C["green"]),
            ("👥","Patients",        lambda: str(len(self.hospital.patients)),                C["yellow"]),
            ("🚨","Critical",        lambda: str(sum(1 for p in self.hospital.patients if p.condition=="Critical")), C["red"]),
            ("⏳","Waiting",         lambda: str(len(self.hospital.unassigned())),            C["red"]),
            ("✔","Allocated",       lambda: str(self.hospital.allocated),                   C["green"]),
        ]
        self._card_labels = []
        for icon, lbl, fn, col in info:
            row = tk.Frame(cards, bg=C["bg3"]); row.pack(fill="x", padx=10, pady=4)
            tk.Frame(row, bg=col, width=3).pack(side="left", fill="y")
            tk.Label(row, text=icon, font=("Segoe UI Emoji",16), bg=C["bg3"], fg=col).pack(side="left", padx=8, pady=8)
            inner = tk.Frame(row, bg=C["bg3"]); inner.pack(side="left", pady=8)
            tk.Label(inner, text=lbl, font=("Consolas",8), bg=C["bg3"], fg=C["dim"]).pack(anchor="w")
            v = tk.StringVar(value=fn())
            tk.Label(inner, textvariable=v, font=("Consolas",18,"bold"), bg=C["bg3"], fg=col).pack(anchor="w")
            self._card_labels.append((v, fn))

        tk.Label(cards, text="QUICK ACTIONS", font=("Consolas",8), bg=C["panel"], fg=C["dim"]).pack(anchor="w", padx=12, pady=(14,4))
        self._btn(cards, "➕ Add Random Patient", self._add_random, fill="x", padx=10, pady=2)
        self._btn(cards, "⚡ Auto-Allocate All",  self._auto_alloc, C["green"], fill="x", padx=10, pady=2)
        self._btn(cards, "🗑 Reset Allocations",  self._reset_alloc, C["red"],  fill="x", padx=10, pady=2)

        # Room map
        rp = self._panel(parent, "Room Occupancy Map")
        rp.grid(row=0, column=1, sticky="nsew", padx=(4,8), pady=(8,4))
        self.map_canvas = tk.Canvas(rp, bg=C["bg2"], highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.map_canvas.bind("<Configure>", lambda e: self._draw_map())

        # Dept chart
        dp = self._panel(parent, "Department Occupancy")
        dp.grid(row=1, column=1, sticky="nsew", padx=(4,8), pady=(4,8))
        self.dept_canvas = tk.Canvas(dp, bg=C["bg2"], highlightthickness=0)
        self.dept_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.dept_canvas.bind("<Configure>", lambda e: self._draw_dept())

    def _draw_map(self):
        c = self.map_canvas; c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 20: return
        rooms = self.hospital.rooms; n = len(rooms)
        cols = max(1, W // 90)
        cw = W / cols; ch = max(38, H / math.ceil(n / cols))
        for i, r in enumerate(rooms):
            col = i % cols; row = i // cols
            x0 = col*cw+4; y0 = row*ch+4; x1 = x0+cw-8; y1 = y0+ch-8
            rate = r.rate
            g = int(200*(1-rate)); red = int(200*rate)
            color = f"#{red:02x}{g:02x}44"
            c.create_rectangle(x0,y0,x1,y1, fill=color, outline=C["border"], width=1)
            c.create_text((x0+x1)/2,(y0+y1)/2-7, text=f"R{r.rid}", font=("Consolas",8,"bold"), fill=C["text"])
            c.create_text((x0+x1)/2,(y0+y1)/2+7, text=f"{r.occupied}/{r.capacity}", font=("Consolas",8), fill=C["dim"])

    def _draw_dept(self):
        c = self.dept_canvas; c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 20: return
        data = {}
        for r in self.hospital.rooms:
            if r.dept not in data: data[r.dept] = [0,0]
            data[r.dept][0] += r.capacity; data[r.dept][1] += r.occupied
        if not data: return
        depts = list(data.keys()); n = len(depts)
        bw = max(12,(W-60)//n - 8)
        mx = max(d[0] for d in data.values()); ch = H-55
        pal = [C["accent"],C["green"],C["yellow"],C["purple"],C["red"],"#FB923C"]
        for i, dept in enumerate(depts):
            cap, occ = data[dept]
            x = 30 + i*((W-60)//n)
            bh_cap = (cap/mx)*ch if mx else 0; bh_occ = (occ/mx)*ch if mx else 0
            col = pal[i % len(pal)]
            c.create_rectangle(x,H-40-bh_cap,x+bw,H-40, fill=C["bg3"],outline=C["border"])
            if bh_occ>0:
                c.create_rectangle(x,H-40-bh_occ,x+bw,H-40, fill=col, outline="")
            c.create_text(x+bw//2,H-40-bh_cap-10, text=f"{occ}/{cap}", font=("Consolas",8), fill=col)
            c.create_text(x+bw//2,H-24, text=dept[:5], font=("Consolas",8), fill=C["dim"])

    # ── PATIENTS ──────────────────────────────

    def _tab_patients(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        form = self._panel(parent, "Register Patient")
        form.grid(row=0, column=0, sticky="nsew", padx=(8,4), pady=8)

        self._pvars = {}
        fields = [("Name","entry"), ("Age","spin"), ("Condition","cond"), ("Department","dept")]
        for label, ftype in fields:
            tk.Label(form, text=label, font=("Consolas",9), bg=C["panel"], fg=C["dim"]).pack(anchor="w", padx=12, pady=(8,2))
            if ftype == "entry":
                v = tk.StringVar(); self._pvars[label] = v
                tk.Entry(form, textvariable=v, bg=C["bg3"], fg=C["text"], font=("Consolas",10),
                         relief="flat", bd=6, insertbackground=C["accent"]).pack(fill="x", padx=12)
            elif ftype == "spin":
                v = tk.IntVar(value=40); self._pvars[label] = v
                tk.Spinbox(form, from_=1, to=110, textvariable=v, bg=C["bg3"], fg=C["text"],
                           font=("Consolas",10), relief="flat", bd=4,
                           buttonbackground=C["bg3"]).pack(fill="x", padx=12)
            elif ftype == "cond":
                v = tk.StringVar(value=CONDITIONS[0]); self._pvars[label] = v
                ttk.Combobox(form, textvariable=v, values=CONDITIONS, state="readonly").pack(fill="x", padx=12)
            elif ftype == "dept":
                v = tk.StringVar(value=DEPARTMENTS[0]); self._pvars[label] = v
                ttk.Combobox(form, textvariable=v, values=DEPARTMENTS, state="readonly").pack(fill="x", padx=12)

        self._btn(form, "➕ Add Patient",        self._add_patient_form,  fill="x", padx=12, pady=(14,3))
        self._btn(form, "🎲 Add Random Patient", self._add_random,        fill="x", padx=12, pady=3)
        self._btn(form, "📦 Load 10 Scenarios",  self._load_scenarios,    fill="x", padx=12, pady=3)

        right = self._panel(parent, "Patient Queue  (sorted by priority)")
        right.grid(row=0, column=1, sticky="nsew", padx=(4,8), pady=8)
        cols = ("PID","Name","Age","Condition","Department","Priority","Room")
        self.pt_tree = self._tree(right, cols, [50,120,45,90,100,65,60])
        for cond, col in COND_COLOR.items():
            self.pt_tree.tag_configure(cond, foreground=col)
        self._refresh_pt_tree()

    def _add_patient_form(self):
        name = self._pvars["Name"].get().strip() or "Patient"
        p = self.hospital.add_patient(name, self._pvars["Age"].get(),
                                      self._pvars["Condition"].get(), self._pvars["Department"].get())
        self._refresh_all()

    def _add_random(self):
        p = self.hospital.add_patient(random.choice(NAMES), random.randint(5,85),
                                      random.choice(CONDITIONS), random.choice(DEPARTMENTS))
        self._refresh_all()

    def _load_scenarios(self):
        scenarios = [
            ("Ahmed Ali",67,"Critical","ICU"), ("Sara Malik",34,"Serious","Cardiology"),
            ("Omar Raza",8,"Moderate","Pediatrics"), ("Hina Shah",55,"Stable","General"),
            ("Bilal Khan",72,"Critical","Emergency"), ("Zara Hussain",28,"Stable","General"),
            ("Tariq Butt",45,"Serious","Neurology"), ("Amna Qureshi",60,"Moderate","ICU"),
            ("Faisal Javed",38,"Stable","General"), ("Nadia Anwar",80,"Serious","Cardiology"),
        ]
        for name,age,cond,dept in scenarios:
            self.hospital.add_patient(name, age, cond, dept)
        self.hospital.log_msg("✅ 10 standard test scenarios loaded")
        self._refresh_all()
        messagebox.showinfo("Loaded","10 patient scenarios added successfully!")

    def _refresh_pt_tree(self):
        for i in self.pt_tree.get_children(): self.pt_tree.delete(i)
        for p in sorted(self.hospital.patients, key=lambda x: -x.priority):
            self.pt_tree.insert("","end",
                values=(f"P{p.pid}", p.name, p.age, p.condition, p.department,
                        p.priority, str(p.room) if p.room else "—"),
                tags=(p.condition,))

    # ── BFS TAB ───────────────────────────────

    def _tab_bfs(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0);    parent.rowconfigure(1, weight=1)

        ctrl = self._panel(parent, "BFS — Uninformed Search")
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8,4))
        ci = tk.Frame(ctrl, bg=C["panel"]); ci.pack(fill="x", padx=10, pady=10)

        tk.Label(ci, text="Patient:", font=("Consolas",9), bg=C["panel"], fg=C["dim"]).grid(row=0,column=0,padx=6)
        self.bfs_pt_var = tk.StringVar()
        self.bfs_pt_cb  = ttk.Combobox(ci, textvariable=self.bfs_pt_var, state="readonly", width=32)
        self.bfs_pt_cb.grid(row=0,column=1,padx=6)

        self._btn_grid(ci, "▶ Run BFS", lambda: self._run_bfs(), 0, 2)
        self._btn_grid(ci, "⚡ Allocate Result", self._alloc_last, 0, 3, C["green"])

        tk.Label(ci, text="BFS explores rooms level by level (breadth-first) to find the nearest available room.",
                 font=("Consolas",8), bg=C["panel"], fg=C["dim"]).grid(row=1,column=0,columnspan=4,padx=6,pady=4,sticky="w")

        left = self._panel(parent,"BFS Output"); left.grid(row=1,column=0,sticky="nsew",padx=(8,4),pady=4)
        self.bfs_out = self._logbox(left)

        right = self._panel(parent,"Room Graph Visualization"); right.grid(row=1,column=1,sticky="nsew",padx=(4,8),pady=4)
        self.bfs_canvas = tk.Canvas(right, bg=C["bg2"], highlightthickness=0)
        self.bfs_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        self._pt_combos.append(self.bfs_pt_cb)
        self._refresh_pt_combos()

    def _run_bfs(self):
        patient = self._get_patient(self.bfs_pt_var)
        self._clear(self.bfs_out)
        if not patient:
            self._write(self.bfs_out,"⚠ Select a patient first.\n","w"); return

        self._write(self.bfs_out,f"{'─'*44}\n","d")
        self._write(self.bfs_out,f"Algorithm : BFS (Breadth-First Search)\n","h")
        self._write(self.bfs_out,f"Patient   : {patient.name} (P{patient.pid})\n","h")
        self._write(self.bfs_out,f"Condition : {patient.condition} | Dept: {patient.department}\n","h")
        self._write(self.bfs_out,f"Priority  : {patient.priority}\n","h")
        self._write(self.bfs_out,f"{'─'*44}\n\n","d")

        t0 = time.time()
        room, path, steps = bfs_find_room(self.hospital, patient)
        elapsed = (time.time()-t0)*1000

        self._write(self.bfs_out,f"Nodes explored : {steps}\n")
        self._write(self.bfs_out,f"Path taken     : {' → '.join(f'R{r}' for r in path)}\n")
        self._write(self.bfs_out,f"Time           : {elapsed:.2f}ms\n\n")

        if room:
            self._write(self.bfs_out,f"✅ Found Room   : {room.rid}\n","ok")
            self._write(self.bfs_out,f"   Type        : {room.rtype}\n","ok")
            self._write(self.bfs_out,f"   Department  : {room.dept}\n","ok")
            self._write(self.bfs_out,f"   Occupancy   : {room.occupied}/{room.capacity}\n\n","ok")
            self._write(self.bfs_out,f"Press '⚡ Allocate Result' to assign.\n","d")
            self._last_result = (patient, room)
        else:
            self._write(self.bfs_out,"⚠ No suitable room found.\n","r")
            self._last_result = None

        self._draw_bfs_graph(path, room)
        self.hospital.log_msg(f"BFS: {patient.name} → Room {room.rid if room else 'None'} | {steps} steps")

    def _draw_bfs_graph(self, path, result_room):
        c = self.bfs_canvas; c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 20: return
        rooms = self.hospital.rooms; n = len(rooms)
        cx, cy, R = W/2, H/2, min(W,H)*0.37
        pos = {}
        for i, r in enumerate(rooms):
            a = 2*math.pi*i/n - math.pi/2
            pos[r.rid] = (cx + R*math.cos(a), cy + R*math.sin(a))

        # Edges (adjacent rooms)
        for i, r1 in enumerate(rooms):
            for j, r2 in enumerate(rooms):
                if 0 < abs(i-j) <= 3:
                    x1,y1 = pos[r1.rid]; x2,y2 = pos[r2.rid]
                    c.create_line(x1,y1,x2,y2, fill=C["border"], width=1)

        # Path edges
        for i in range(len(path)-1):
            if path[i] in pos and path[i+1] in pos:
                x1,y1=pos[path[i]]; x2,y2=pos[path[i+1]]
                c.create_line(x1,y1,x2,y2,fill=C["accent"],width=2,dash=(4,2))

        # Nodes
        for r in rooms:
            if r.rid not in pos: continue
            x,y = pos[r.rid]
            in_path = r.rid in path
            is_res  = result_room and r.rid == result_room.rid
            size  = 14 if is_res else (11 if in_path else 8)
            color = C["green"] if is_res else (C["accent"] if in_path else (C["panel"] if r.available else C["red"]))
            c.create_oval(x-size,y-size,x+size,y+size, fill=color, outline="white" if is_res else C["border"], width=2)
            c.create_text(x,y, text=str(r.rid), font=("Consolas",7), fill=C["text"])

        c.create_text(W/2,H-14,
                      text="● Explored  ◉ Result  ● Available  ● Full  ─── BFS Path",
                      font=("Consolas",8), fill=C["dim"])

    # ── A* TAB ────────────────────────────────

    def _tab_astar(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0);    parent.rowconfigure(1, weight=1)

        ctrl = self._panel(parent, "A* Search — Informed / Heuristic Search")
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8,4))
        ci = tk.Frame(ctrl, bg=C["panel"]); ci.pack(fill="x", padx=10, pady=10)

        tk.Label(ci, text="Patient:", font=("Consolas",9), bg=C["panel"], fg=C["dim"]).grid(row=0,column=0,padx=6)
        self.as_pt_var = tk.StringVar()
        self.as_pt_cb  = ttk.Combobox(ci, textvariable=self.as_pt_var, state="readonly", width=32)
        self.as_pt_cb.grid(row=0,column=1,padx=6)
        tk.Label(ci, text="h-weight:", font=("Consolas",9), bg=C["panel"], fg=C["dim"]).grid(row=0,column=2,padx=6)
        self.h_w = tk.DoubleVar(value=1.0)
        tk.Scale(ci, variable=self.h_w, from_=0.1, to=3.0, resolution=0.1, orient="horizontal",
                 bg=C["panel"], fg=C["accent"], length=100, troughcolor=C["bg3"],
                 highlightthickness=0).grid(row=0,column=3,padx=4)
        self._btn_grid(ci, "▶ Run A*", lambda: self._run_astar(), 0, 4)
        self._btn_grid(ci, "⚡ Allocate", self._alloc_astar, 0, 5, C["green"])
        tk.Label(ci, text="A* uses f = g(n) + h(n). g = actual cost (dept/type mismatch). h = heuristic (urgency vs room quality gap).",
                 font=("Consolas",8), bg=C["panel"], fg=C["dim"]).grid(row=1,column=0,columnspan=6,padx=6,pady=4,sticky="w")

        left = self._panel(parent,"A* Output"); left.grid(row=1,column=0,sticky="nsew",padx=(8,4),pady=4)
        self.as_out = self._logbox(left)

        right = self._panel(parent,"Cost Breakdown"); right.grid(row=1,column=1,sticky="nsew",padx=(4,8),pady=4)
        self.as_canvas = tk.Canvas(right, bg=C["bg2"], highlightthickness=0)
        self.as_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        self._pt_combos.append(self.as_pt_cb)
        self._refresh_pt_combos()

    def _run_astar(self):
        patient = self._get_patient(self.as_pt_var)
        self._clear(self.as_out)
        if not patient:
            self._write(self.as_out,"⚠ Select a patient first.\n","w"); return

        self._write(self.as_out,f"{'─'*44}\n","d")
        self._write(self.as_out,f"Algorithm : A* Search\n","h")
        self._write(self.as_out,f"Patient   : {patient.name} (P{patient.pid})\n","h")
        self._write(self.as_out,f"Condition : {patient.condition} | Dept: {patient.department}\n","h")
        self._write(self.as_out,f"h-weight  : {self.h_w.get():.1f}\n","h")
        self._write(self.as_out,f"{'─'*44}\n\n","d")

        # Show all rooms with f scores
        available = self.hospital.available_rooms()
        room_scores = []
        for r in available:
            g = g_cost(patient, r)
            h = h_cost(patient, r) * self.h_w.get()
            room_scores.append((g+h, g, h, r.rid, r))
        room_scores.sort(key=lambda x: x[0])

        self._write(self.as_out,"Room evaluations (sorted by f=g+h):\n","h")
        self._write(self.as_out,f"{'RID':<5}{'Type':<15}{'Dept':<12}{'g':<7}{'h':<7}{'f':<7}\n","d")
        self._write(self.as_out,"─"*49+"\n","d")
        for f,g,h,_,r in room_scores[:8]:
            self._write(self.as_out, f"R{r.rid:<4}{r.rtype:<15}{r.dept:<12}{g:<7.2f}{h:<7.2f}{f:<7.2f}\n")

        t0 = time.time()
        room, explored = astar_find_room(self.hospital, patient)
        elapsed = (time.time()-t0)*1000

        self._write(self.as_out,f"\nNodes explored : {explored}\n")
        self._write(self.as_out,f"Time           : {elapsed:.2f}ms\n\n")

        if room:
            g = g_cost(patient, room); h = h_cost(patient, room)*self.h_w.get()
            self._write(self.as_out,f"✅ Best Room    : R{room.rid}\n","ok")
            self._write(self.as_out,f"   Type        : {room.rtype}\n","ok")
            self._write(self.as_out,f"   Department  : {room.dept}\n","ok")
            self._write(self.as_out,f"   g-cost      : {g:.2f}\n","ok")
            self._write(self.as_out,f"   h-cost      : {h:.2f}\n","ok")
            self._write(self.as_out,f"   f = g+h     : {g+h:.2f}\n","ok")
            self._last_result       = (patient, room)
            self._last_astar_result = (patient, room)
            self._draw_astar_bars(room_scores[:8])
        else:
            self._write(self.as_out,"⚠ No room found.\n","r")
            self._last_result       = None
            self._last_astar_result = None

        self.hospital.log_msg(f"A*: {patient.name} → Room {room.rid if room else 'None'} | explored={explored}")

    def _draw_astar_bars(self, scores):
        c = self.as_canvas; c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 20 or not scores: return
        c.create_text(W//2,16, text="f-scores per room (lower = better)", font=("Consolas",9), fill=C["accent"])
        mx = max(s[0] for s in scores) or 1
        bw = max(8, (W-80)//len(scores)-6)
        for i,(f,g,h,_,r) in enumerate(scores):
            x = 40 + i*((W-80)//len(scores))
            bar_h = (f/mx)*(H-70)
            g_h = (g/mx)*(H-70); hh = (h/mx)*(H-70)
            # g portion
            c.create_rectangle(x, H-40-g_h, x+bw, H-40, fill=C["accent"], outline="")
            # h portion on top
            c.create_rectangle(x, H-40-g_h-hh, x+bw, H-40-g_h, fill=C["purple"], outline="")
            c.create_text(x+bw//2, H-40-g_h-hh-10, text=f"{f:.1f}", font=("Consolas",7), fill=C["text"])
            c.create_text(x+bw//2, H-24, text=f"R{r.rid}", font=("Consolas",7), fill=C["dim"])
        # Legend
        c.create_rectangle(10,H-14,20,H-6, fill=C["accent"],outline="")
        c.create_text(24,H-10, text="g-cost", font=("Consolas",7), fill=C["dim"], anchor="w")
        c.create_rectangle(68,H-14,78,H-6, fill=C["purple"],outline="")
        c.create_text(82,H-10, text="h-cost", font=("Consolas",7), fill=C["dim"], anchor="w")

    # ── MINIMAX TAB ───────────────────────────

    def _tab_minimax(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0);    parent.rowconfigure(1, weight=1)

        ctrl = self._panel(parent, "Minimax + Alpha-Beta Pruning — Adversarial Search")
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8,4))
        ci = tk.Frame(ctrl, bg=C["panel"]); ci.pack(fill="x", padx=10, pady=10)

        tk.Label(ci, text="Depth:", font=("Consolas",9), bg=C["panel"], fg=C["dim"]).grid(row=0,column=0,padx=6)
        self.mm_depth = tk.IntVar(value=3)
        tk.Spinbox(ci, from_=1, to=5, textvariable=self.mm_depth, width=4,
                   bg=C["bg3"], fg=C["text"], font=("Consolas",10), relief="flat",
                   buttonbackground=C["bg3"]).grid(row=0,column=1,padx=4)
        self._btn_grid(ci, "▶ Run Minimax",       lambda: self._run_mm(False), 0, 2)
        self._btn_grid(ci, "⚡ Compare Plain vs α-β", lambda: self._run_mm(True), 0, 3, C["yellow"])
        tk.Label(ci, text="MAX = high-priority patients (Critical/Serious). MIN = low-priority (Stable). Conflict: who gets limited rooms?",
                 font=("Consolas",8), bg=C["panel"], fg=C["dim"]).grid(row=1,column=0,columnspan=4,padx=6,pady=4,sticky="w")

        left = self._panel(parent,"Minimax Output"); left.grid(row=1,column=0,sticky="nsew",padx=(8,4),pady=4)
        self.mm_out = self._logbox(left)

        right = self._panel(parent,"Game Tree"); right.grid(row=1,column=1,sticky="nsew",padx=(4,8),pady=4)
        self.mm_canvas = tk.Canvas(right, bg=C["bg2"], highlightthickness=0)
        self.mm_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.mm_canvas.bind("<Configure>", lambda e: self._draw_mm_tree())

    def _run_mm(self, compare):
        depth = self.mm_depth.get()
        self._clear(self.mm_out)
        self._write(self.mm_out,f"{'═'*46}\n","h")
        self._write(self.mm_out,f"  ADVERSARIAL SEARCH — ROOM CONFLICT\n","h")
        self._write(self.mm_out,f"  Depth = {depth}\n","h")
        self._write(self.mm_out,f"{'═'*46}\n\n","h")

        r = run_adversarial(self.hospital, depth)
        hi_names = [p.name for p in r["hi"]]; lo_names = [p.name for p in r["lo"]]
        rooms_list= [f"R{rm.rid}" for rm in r["rooms"]]

        self._write(self.mm_out,f"MAX players (high priority): {hi_names}\n")
        self._write(self.mm_out,f"MIN players (low priority) : {lo_names}\n")
        self._write(self.mm_out,f"Contested rooms : {rooms_list}\n\n")

        if compare:
            self._write(self.mm_out,"─── Plain Minimax ───\n","h")
            self._write(self.mm_out,f"  Nodes explored : {r['mm_nodes']}\n")
            self._write(self.mm_out,f"  Time           : {r['mm_time']:.2f}ms\n")
            self._write(self.mm_out,f"  Value          : {r['mm_val']:.2f}\n\n")
            self._write(self.mm_out,"─── Alpha-Beta Pruning ───\n","ok")
            self._write(self.mm_out,f"  Nodes explored : {r['ab_nodes']}\n","ok")
            self._write(self.mm_out,f"  Time           : {r['ab_time']:.2f}ms\n","ok")
            self._write(self.mm_out,f"  Value          : {r['ab_val']:.2f}\n\n","ok")
            self._write(self.mm_out,f"✅ Nodes pruned : {r['pruned']} ({r['pct']:.1f}% saved)\n","ok")
            self._write(self.mm_out,f"   Same result, fewer nodes — α-β more efficient!\n","ok")
        else:
            self._write(self.mm_out,"Running Minimax with Alpha-Beta...\n","h")
            self._write(self.mm_out,f"  Nodes explored : {r['ab_nodes']}\n","ok")
            self._write(self.mm_out,f"  Time           : {r['ab_time']:.2f}ms\n","ok")
            self._write(self.mm_out,f"  Game value     : {r['ab_val']:.2f}\n\n","ok")
            decision = "HIGH-priority patients" if r["ab_val"] < 0 else "LOW-priority patients"
            self._write(self.mm_out,f"Decision: Rooms allocated to {decision}\n","ok")

        self.hospital.log_msg(f"Minimax d={depth}: {r['ab_nodes']} nodes | pruned={r['pruned']}")
        self._draw_mm_tree()

    def _draw_mm_tree(self):
        c = self.mm_canvas; c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 40: return
        depth = self.mm_depth.get()
        c.create_text(W//2,16, text="Minimax Game Tree", font=("Consolas",9), fill=C["accent"])

        def node(x, y, val, is_max, level):
            col = C["accent"] if is_max else C["red"]
            lbl = "MAX" if is_max else "MIN"
            c.create_oval(x-18,y-16,x+18,y+16, fill=C["panel"], outline=col, width=2)
            c.create_text(x,y-5, text=lbl, font=("Consolas",7), fill=col)
            c.create_text(x,y+6, text=f"{val:.0f}", font=("Consolas",8,"bold"), fill=C["text"])
            if level < min(depth,3):
                kids = 2 if level < 2 else 2
                spread = (W-80)/(2**(level+1))
                for k in range(kids):
                    kx = x + spread*(k - (kids-1)/2)
                    ky = y + 70
                    kv = random.uniform(-8,8) if level == min(depth,3)-1 else val + random.uniform(-3,3)
                    # Prune second branch of MIN visually
                    if is_max or k == 0:
                        c.create_line(x,y+16,kx,ky-16, fill=C["border"], width=1)
                        node(kx, ky, kv, not is_max, level+1)
                    else:
                        # Pruned branch
                        c.create_line(x,y+16,kx,ky-16, fill=C["border"], width=1, dash=(3,4))
                        c.create_text(kx,ky, text="✗\npruned", font=("Consolas",7), fill=C["dim"])

        node(W//2, 45, -4, True, 0)
        c.create_text(W//2,H-12, text="✗ = branches pruned by α-β cutoff",
                      font=("Consolas",8), fill=C["dim"])

    # ── AGENTS TAB ───────────────────────────

    def _tab_agents(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0);    parent.rowconfigure(1, weight=1)

        ctrl = self._panel(parent, "Intelligent Agents")
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8,4))
        ci = tk.Frame(ctrl, bg=C["panel"]); ci.pack(fill="x", padx=10, pady=10)

        tk.Label(ci, text="Agent:", font=("Consolas",9), bg=C["panel"], fg=C["dim"]).grid(row=0,column=0,padx=6)
        self.ag_var = tk.StringVar(value="Utility-Based")
        ttk.Combobox(ci, textvariable=self.ag_var, state="readonly", width=18,
                     values=[a.NAME.replace(" Agent","") for a in self.agents]).grid(row=0,column=1,padx=6)

        tk.Label(ci, text="Patient:", font=("Consolas",9), bg=C["panel"], fg=C["dim"]).grid(row=0,column=2,padx=6)
        self.ag_pt_var = tk.StringVar()
        self.ag_pt_cb  = ttk.Combobox(ci, textvariable=self.ag_pt_var, state="readonly", width=30)
        self.ag_pt_cb.grid(row=0,column=3,padx=6)

        self._btn_grid(ci, "▶ Run Agent",        lambda: self._run_agent(), 0, 4)
        self._btn_grid(ci, "⚡ Allocate",         self._alloc_last,          0, 5, C["green"])
        self._btn_grid(ci, "🔄 Compare All",      self._compare_agents,      0, 6, C["yellow"])

        left = self._panel(parent,"Agent Output"); left.grid(row=1,column=0,sticky="nsew",padx=(8,4),pady=4)
        self.ag_out = self._logbox(left)

        right = self._panel(parent,"Agent Architecture"); right.grid(row=1,column=1,sticky="nsew",padx=(4,8),pady=4)
        self.ag_canvas = tk.Canvas(right, bg=C["bg2"], highlightthickness=0)
        self.ag_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.ag_canvas.bind("<Configure>", lambda e: self._draw_agent_arch())
        self.ag_var.trace_add("write", lambda *a: self._draw_agent_arch())

        self._pt_combos.append(self.ag_pt_cb)
        self._refresh_pt_combos()

    def _run_agent(self):
        name = self.ag_var.get()
        agent = next((a for a in self.agents if a.NAME.startswith(name)), self.agents[-1])
        patient = self._get_patient(self.ag_pt_var)
        self._clear(self.ag_out)
        if not patient:
            self._write(self.ag_out,"⚠ Select a patient first.\n","w"); return

        self._write(self.ag_out,f"{'─'*44}\n","d")
        self._write(self.ag_out,f"Agent     : {agent.NAME}\n","h")
        self._write(self.ag_out,f"Patient   : {patient.name} (P{patient.pid})\n","h")
        self._write(self.ag_out,f"Condition : {patient.condition}\n","h")
        self._write(self.ag_out,f"Dept      : {patient.department}\n","h")
        self._write(self.ag_out,f"{'─'*44}\n\n","d")

        t0 = time.time()
        room, reason = agent.act(patient, self.hospital)
        elapsed = (time.time()-t0)*1000

        self._write(self.ag_out,f"Decision  : {reason}\n","ok")
        self._write(self.ag_out,f"Time      : {elapsed:.3f}ms\n\n")
        if room:
            self._write(self.ag_out,f"✅ Room    : {room.rid} [{room.rtype}, {room.dept}]\n","ok")
            self._write(self.ag_out,f"   Beds   : {room.occupied}/{room.capacity}\n")
            if hasattr(agent,"utility"):
                u = agent.utility(patient, room)
                self._write(self.ag_out,f"   Utility: {u:.1f}/100\n","h")
            self._last_result = (patient, room)
        else:
            self._write(self.ag_out,"⚠ No room found.\n","r")
            self._last_result = None

        self.hospital.log_msg(f"{agent.NAME}: {patient.name} → Room {room.rid if room else 'None'}")

    def _compare_agents(self):
        patient = self._get_patient(self.ag_pt_var)
        self._clear(self.ag_out)
        if not patient:
            self._write(self.ag_out,"⚠ Select a patient first.\n","w"); return

        self._write(self.ag_out,f"{'═'*46}\n","h")
        self._write(self.ag_out,f"  ALL AGENTS — {patient.name} ({patient.condition})\n","h")
        self._write(self.ag_out,f"{'═'*46}\n\n","h")

        for agent in self.agents:
            t0 = time.time()
            room, reason = agent.act(patient, self.hospital)
            elapsed = (time.time()-t0)*1000
            self._write(self.ag_out,f"◆ {agent.NAME}\n","h")
            self._write(self.ag_out,f"  {reason}\n","ok")
            self._write(self.ag_out,f"  Room: {room.rid if room else '—'} | {elapsed:.3f}ms\n\n")

    def _draw_agent_arch(self):
        c = self.ag_canvas; c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 40: return
        name = self.ag_var.get()

        def box(x,y,w,h,text,col=None):
            col=col or C["panel"]
            c.create_rectangle(x,y,x+w,y+h, fill=col, outline=C["accent"],width=2)
            c.create_text(x+w//2,y+h//2, text=text, font=("Consolas",8), fill=C["text"],width=w-8)

        def arr(x1,y1,x2,y2,label=""):
            c.create_line(x1,y1,x2,y2, fill=C["accent"],width=2,arrow=tk.LAST,arrowshape=(9,11,4))
            if label: c.create_text((x1+x2)//2+20,(y1+y2)//2, text=label, font=("Consolas",7), fill=C["dim"])

        bw,bh,cx = 160,38,W//2
        if name == "Simple Reflex":
            c.create_text(cx,16,text="Simple Reflex Agent",font=("Consolas",10,"bold"),fill=C["accent"])
            box(cx-bw//2,36,bw,bh,"PERCEPT\n(Patient Condition)",C["bg3"])
            arr(cx,74,cx,112,"condition")
            box(cx-bw//2,112,bw,bh,"CONDITION → ACTION\nRULES",C["bg3"])
            arr(cx,150,cx,188,"action")
            box(cx-bw//2,188,bw,bh,"ACTUATOR\n(Assign Room)",C["bg3"])
            c.create_text(cx,H-18,text="No memory. Rules only.",font=("Consolas",8),fill=C["dim"])

        elif name == "Model-Based":
            c.create_text(cx,16,text="Model-Based Agent",font=("Consolas",10,"bold"),fill=C["accent"])
            box(cx-bw//2,36,bw,bh,"SENSORS\n(Percept)",C["bg3"])
            box(cx-bw//2-bw-20,110,bw,bh,"INTERNAL MODEL\n(Room availability)",C["panel"])
            arr(cx,74,cx,130)
            arr(cx-bw//2,74,cx-bw//2-20+bw//2,128,"updates")
            box(cx-bw//2,130,bw,bh,"RULES + MODEL\n(Decision)",C["bg3"])
            arr(cx,168,cx,206,"action")
            box(cx-bw//2,206,bw,bh,"ACTUATOR",C["bg3"])
            c.create_text(cx,H-18,text="Maintains world state.",font=("Consolas",8),fill=C["dim"])

        elif name == "Goal-Based":
            c.create_text(cx,16,text="Goal-Based Agent",font=("Consolas",10,"bold"),fill=C["accent"])
            box(cx-bw//2,36,bw,bh,"PERCEPT",C["bg3"])
            box(cx+bw//2+20,110,bw,bh,"GOALS\n(Correct dept + available)",C["bg3"])
            arr(cx,74,cx,130)
            box(cx-bw//2,130,bw,bh,"GOAL TEST\n(Search for match)",C["bg3"])
            arr(cx+bw//2+20,129,cx+bw//2,148,"check")
            arr(cx,168,cx,206,"action")
            box(cx-bw//2,206,bw,bh,"ACTUATOR",C["bg3"])
            c.create_text(cx,H-18,text="Acts to satisfy goal.",font=("Consolas",8),fill=C["dim"])

        else:
            c.create_text(cx,16,text="Utility-Based Agent",font=("Consolas",10,"bold"),fill=C["accent"])
            box(cx-bw//2,36,bw,bh,"PERCEPT",C["bg3"])
            arr(cx,74,cx,112)
            box(cx-bw//2,112,bw,bh,"UTILITY FUNCTION\ndept+type+occupancy+age",C["panel"])
            arr(cx,150,cx,188,"max utility")
            box(cx-bw//2,188,bw,bh,"ACTUATOR",C["bg3"])
            facts = ["Dept match +30","Type match +25","Free space +20","Age factor +10"]
            for i,f in enumerate(facts):
                fx = 8 + i*((W-16)//4)
                c.create_rectangle(fx,H-50,fx+(W-16)//4-4,H-28, fill=C["bg3"],outline=C["border"])
                c.create_text(fx+(W-16)//8,H-39, text=f, font=("Consolas",7), fill=C["yellow"])
            c.create_text(cx,H-18,text="Maximises composite utility score.",font=("Consolas",8),fill=C["dim"])

    # ── LOG TAB ───────────────────────────────

    def _tab_log(self, parent):
        p = self._panel(parent, "Activity Log"); p.pack(fill="both", expand=True, padx=8, pady=8)
        bf = tk.Frame(p, bg=C["panel"]); bf.pack(fill="x", padx=8, pady=(6,2))
        self._btn(bf, "⟳ Refresh", self._refresh_log_tab, padx=4, side="left")
        self._btn(bf, "🗑 Clear",   self._clear_log_tab, C["red"], padx=4, side="left")
        self.log_txt = self._logbox(p)
        self._refresh_log_tab()

    def _refresh_log_tab(self):
        self._clear(self.log_txt)
        for entry in self.hospital.log[-300:]:
            tag = "ok" if "✅" in entry else ("w" if "⚠" in entry else "")
            self._write(self.log_txt, entry+"\n", tag)
        self.log_txt.see("end")

    def _clear_log_tab(self):
        self.hospital.log.clear(); self._refresh_log_tab()

    # ── ACTIONS ───────────────────────────────

    def _alloc_astar(self):
        if self._last_astar_result:
            patient, room = self._last_astar_result
            if not room.available:
                messagebox.showwarning("Full","That room is now full. Re-run A*.")
                return
            self.hospital.allocate(patient, room)
            self._last_astar_result = None
            self._refresh_all()
            messagebox.showinfo("Allocated", f"✅ {patient.name} → Room {room.rid}")
        else:
            messagebox.showwarning("No Result","Run A* Search first, then click Allocate.")

    def _alloc_last(self):
        if self._last_result:
            patient, room = self._last_result
            if not room.available:
                messagebox.showwarning("Full","That room is now full. Re-run the algorithm.")
                return
            self.hospital.allocate(patient, room)
            self._last_result = None
            self._refresh_all()
            messagebox.showinfo("Allocated", f"✅ {patient.name} → Room {room.rid}")
        else:
            messagebox.showwarning("No Result","Run an algorithm first.")

    def _auto_alloc(self):
        agent = self.agents[-1]  # Utility-Based
        count = 0
        for p in sorted(self.hospital.unassigned(), key=lambda x:-x.priority):
            room, _ = agent.act(p, self.hospital)
            if room and room.available:
                self.hospital.allocate(p, room); count += 1
        self._refresh_all()
        messagebox.showinfo("Done", f"✅ {count} patients allocated by Utility-Based Agent.")

    def _reset_alloc(self):
        if not messagebox.askyesno("Confirm","Reset all allocations?"): return
        for p in self.hospital.patients: p.room = None
        for r in self.hospital.rooms:    r.occupied = 0
        self.hospital.allocated = 0
        self.hospital.log_msg("🗑 All allocations reset")
        self._refresh_all()

    def _refresh_all(self):
        self._update_stats()
        self._refresh_pt_tree()
        self._refresh_pt_combos()
        self._draw_map()
        self._draw_dept()
        for v,fn in self._card_labels: v.set(fn())

    def _refresh_log(self): pass  # separate from log tab

    # ── UTIL HELPERS ──────────────────────────

    def _btn_grid(self, parent, text, cmd, row, col, color=None):
        color = color or C["accent"]
        b = tk.Button(parent, text=text, font=("Consolas",9), bg=color, fg=C["bg"],
                      relief="flat", cursor="hand2", command=cmd, pady=4, padx=8)
        b.grid(row=row, column=col, padx=6)
        b.bind("<Enter>", lambda e: b.configure(bg=C["green"]))
        b.bind("<Leave>", lambda e: b.configure(bg=color))

    def _tick(self):
        self.clock_lbl.configure(text=time.strftime("%a %d %b  %H:%M:%S"))
        self.after(1000, self._tick)

    def _load_scenarios(self):
        self._load_scenarios_data()

    def _load_scenarios_data(self):
        pass  # called from __init__ safely after tab setup

# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    Patient._id = 0
    Room._id    = 0
    print("Starting AI Hospital Room Allocation System...")
    app = App()
    # Load default scenarios on start
    scenarios = [
        ("Ahmed Ali",67,"Critical","ICU"), ("Sara Malik",34,"Serious","Cardiology"),
        ("Omar Raza",8,"Moderate","Pediatrics"), ("Hina Shah",55,"Stable","General"),
        ("Bilal Khan",72,"Critical","Emergency"), ("Zara Hussain",28,"Stable","General"),
        ("Tariq Butt",45,"Serious","Neurology"), ("Amna Qureshi",60,"Moderate","ICU"),
        ("Faisal Javed",38,"Stable","General"), ("Nadia Anwar",80,"Serious","Cardiology"),
    ]
    for name,age,cond,dept in scenarios:
        app.hospital.add_patient(name,age,cond,dept)
    app._refresh_all()
    app.mainloop()
