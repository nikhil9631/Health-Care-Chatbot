# CLAUDE.md

## Project

Health Care Chatbot built using Python, Tkinter GUI, and a trained NLP model (intents-based).

---

## Key Files

* `gui.py` — main GUI application (entry point)
* `chatbot.py` — chatbot logic and response generation
* `training.py` — model training script
* `intents.json` — intents and responses dataset
* `chatbotmodel.h5` — trained model
* `words.pkl`, `classes.pkl` — preprocessing data

---

## Setup & Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run application

```bash
py gui.py
```

---

## Code Style

* Use simple, readable Python code (no unnecessary complexity)
* Follow modular design (separate GUI, logic, and training)
* Avoid hardcoding values where possible
* Keep functions small and focused

---

## Do Not

* Do not modify model files (`.h5`, `.pkl`) manually
* Do not remove required dataset files (`intents.json`)
* Do not break GUI structure in `gui.py`
* Do not add heavy dependencies unnecessarily

---

## Testing

* Run chatbot manually via GUI
* Test different user queries from intents.json
* Ensure chatbot responses match expected intents

---

## Workflow

* Make changes in small commits
* Use clear commit messages:

  * "fix: corrected chatbot response logic"
  * "feat: added new intents"
* Always test before pushing

---

## Environment Notes

* Requires Python 3.10+
* Works on Windows (Tkinter GUI)
* Ensure Python is added to PATH

---

## Common Issues

* If GUI does not open → check Python installation
* If model errors → ensure `.h5`, `.pkl` files exist
* If icon error → keep `iconbitmap` line commented

---
