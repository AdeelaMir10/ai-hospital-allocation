# AI Hospital Room Allocation System

This is my semester project for the AI Lab (BS Computer Science). 
The idea is simple — instead of manually assigning hospital rooms, 
the system uses AI search algorithms and agents to do it automatically 
based on how critical the patient is.

Built entirely in Python with a GUI using Tkinter.
---

## What it does

You enter a patient's name, age, condition, and department.
The system then figures out the best available room for them using
different AI algorithms. You can also run all 4 AI agents at once
and compare their decisions.

There's a live dashboard showing room occupancy, patient list,
department stats, and an activity log.

---

## Algorithms I implemented

- **BFS** — searches through rooms like a graph, finds the nearest available one
- **A\* Search** — smarter version, considers department match + room type + occupancy
- **Minimax with Alpha-Beta Pruning** — treats allocation like a game tree
- **4 Types of Agents:**
  - Simple Reflex (just looks at condition and assigns)
  - Model-Based (keeps track of room states)
  - Goal-Based (tries to match department goals)
  - Utility-Based (scores every room and picks the highest)

---

## How priority works

Patients are scored based on condition:

| Condition | Score |
|-----------|-------|
| Critical  | 10    |
| Serious   | 7     |
| Moderate  | 4     |
| Stable    | 1     |

If the patient is over 60, they get +2 added to their score.
Higher score = gets a room first.

---

## How to run it

You just need Python 3 installed. No extra libraries.

```bash
python ai_hospital_allocation.py
```

A window will open with everything inside.

---

## Requirements

- Python 3.8 or above
- Tkinter (already included with Python, nothing to install)

---

## Departments & Room Types

Departments: ICU, General, Cardiology, Pediatrics, Emergency, Neurology

Room types: ICU Room, Private Room, General Ward, Emergency Bay

---

## Notes

- The system loads 10 sample patients automatically when you start it
- You can reset all allocations and start fresh anytime
- The Utility-Based agent has an "Auto Allocate All" button that handles everyone at once

---

## About

Made by Adeela Shaheen Mir — BS Computer Science, AI Lab Project 2026
