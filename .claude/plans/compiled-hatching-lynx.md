# Plan: NLP Accuracy & Reliability Improvements (Phase 2)

## Context

The chatbot's current NN model overfits heavily (95% train / 57.6% val accuracy) because binary bag-of-words loses word importance information. Intent tags have inconsistent casing. Prevention intents have too few patterns. The fallback system needs broader keyword coverage. This phase replaces BoW with TF-IDF, normalizes all tags, expands the dataset, strengthens fallback, and adds comprehensive edge-case tests.

---

## Step 1: Normalize intent tags (`intents.json`)

Rename all tags to `lowercase_underscore` format. 12 tags change:
- `"common cold symptoms"` -> `"common_cold_symptoms"`, `"Diabetes symptoms"` -> `"diabetes_symptoms"`, etc.
- `"Consultation"` -> `"consultation"`
- All prevention tags: spaces -> underscores

This establishes the canonical naming that all other changes depend on.

---

## Step 2: Expand dataset (`intents.json`)

- **Prevention intents**: Add 6-8 patterns each (target 12+), add 1-2 responses each (target 3-4)
  - More conversational: "anything I can take for a headache?", "home remedies for fever"
- **Multi-symptom patterns**: Add 2-3 combo patterns per symptom intent ("headache and fever", "sore throat with sneezing")
- **Vague descriptions**: "I don't feel well", "I feel sick", "something is wrong"
- **Informal/typo patterns**: "i cant breathe", "my head is killin me", "im sneezing alot"
- **Expand `noanswer`**: Add off-topic queries ("what time is it", "play music", "who is the president")
- **Expand `capabilities`**: Add 2 more response variants
- **Target**: ~350 patterns total (up from ~256)

---

## Step 3: Replace BoW with TF-IDF

### 3A: `nlp_utils.py` -- Add `lemmatizing_tokenizer()`
- New function that wraps `tokenize_and_lemmatize()` as a callable for sklearn's `TfidfVectorizer(tokenizer=...)` 
- Ensures identical preprocessing during training fit and inference transform

### 3B: `training_py.py` -- Rewrite vectorization
- Replace manual BoW loop with `TfidfVectorizer(tokenizer=lemmatizing_tokenizer, token_pattern=None)`
- `vectorizer.fit_transform(pattern_strings)` produces TF-IDF matrix directly
- Save `vectorizer.pkl` (replaces `words.pkl`)
- NN architecture stays: Dense(256) -> Dropout(0.5) -> Dense(128) -> Dropout(0.5) -> Dense(n_classes)
- Keep EarlyStopping, validation split, metrics saving

---

## Step 4: Update inference (`chatbot_py.py`)

- Load `vectorizer.pkl` instead of `words.pkl`
- Remove `bag_of_words()` function
- Rewrite `predict_class()`: `vectorizer.transform([sentence]).toarray()` feeds directly to model
- Normalize all KEYWORD_MAP keys to match new tags
- Fix `handle_followup()`: `"_symptoms"` -> `"_prevention"` (underscore format)
- Add empty input guard: `if not message.strip(): return get_fallback_response()`

---

## Step 5: Strengthen fallback system (`chatbot_py.py`)

- **Expand KEYWORD_MAP** values with more synonyms per intent (~5-10 new keywords each)
- **Add `SYMPTOM_SYNONYMS` dict** for informal->formal normalization ("tummy"->"stomach", "bp"->"blood pressure")
- **Add `keyword_match_multi()`** returning all matching intents sorted by score
- **Multi-intent awareness**: When two symptom intents score similarly, acknowledge both in response
- Increase `_recent_responses` buffer from 5 to 8

---

## Step 6: Expand tests (`test_chatbot.py`)

- Update 5 existing assertions for normalized tag names
- **New `TestEdgeCases`**: empty input, very long input, special characters, unicode, numbers-only
- **New `TestConfidenceRouting`**: probability format, normalized tags in predictions, gibberish low confidence
- **New `TestMultiTurnConversation`**: symptom->followup flow, context reset on topic change
- **New `TestResponseContent`**: fallback helpfulness, response variety, all intents have responses
- **New `TestTagNormalization`**: all tags lowercase/underscore, KEYWORD_MAP keys match intents.json, classes.pkl matches

---

## Step 7: Retrain and verify

1. Run `python training_py.py` (generates chatbotmodel.h5, vectorizer.pkl, classes.pkl)
2. Run `python -m unittest test_chatbot -v`
3. Interactive smoke test via `python chatbot_py.py`
4. Manual GUI test via `python gui.py`

---

## Files Modified

| File | Change | Key Modifications |
|------|--------|-------------------|
| `intents.json` | MODIFY | Normalize 12 tags, add ~90 patterns, add ~15 responses |
| `nlp_utils.py` | MODIFY | Add `lemmatizing_tokenizer()` (~3 lines) |
| `training_py.py` | REWRITE | TfidfVectorizer replaces manual BoW, saves vectorizer.pkl |
| `chatbot_py.py` | MODIFY | Load vectorizer, remove bag_of_words(), normalize KEYWORD_MAP, add synonyms, multi-intent detection |
| `test_chatbot.py` | MODIFY | Update tag assertions, add ~30 new tests across 5 new test classes |
| `words.pkl` | DELETE | Replaced by vectorizer.pkl |

**Not modified:** `gui.py` (only calls `chat()`), `requirements.txt` (sklearn already listed)
