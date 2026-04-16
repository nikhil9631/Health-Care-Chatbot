"""Tests for the chatbot NLP pipeline, keyword fallback, and integration."""

import json
import unittest


class TestPreprocessing(unittest.TestCase):
    """Test the shared preprocessing pipeline."""

    def test_tokenize_basic(self):
        from nlp_utils import tokenize_and_lemmatize
        result = tokenize_and_lemmatize("I have a cold")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_tokenize_lowercases(self):
        from nlp_utils import tokenize_and_lemmatize
        result = tokenize_and_lemmatize("FEVER")
        self.assertEqual(result, ["fever"])

    def test_tokenize_removes_punctuation(self):
        from nlp_utils import tokenize_and_lemmatize
        result = tokenize_and_lemmatize("hello?")
        self.assertNotIn("?", result)

    def test_tokenize_lemmatizes(self):
        from nlp_utils import tokenize_and_lemmatize
        result = tokenize_and_lemmatize("running coughing")
        for word in result:
            self.assertEqual(word, word.lower())

    def test_lemmatizing_tokenizer_same_as_tokenize(self):
        from nlp_utils import tokenize_and_lemmatize, lemmatizing_tokenizer
        text = "I have a severe headache and fever"
        self.assertEqual(lemmatizing_tokenizer(text), tokenize_and_lemmatize(text))


class TestKeywordFallback(unittest.TestCase):
    """Test the keyword matching system."""

    def test_keyword_match_cold(self):
        from chatbot_py import keyword_match
        result = keyword_match("I have a runny nose and cough")
        self.assertEqual(result, "common_cold_symptoms")

    def test_keyword_match_fever(self):
        from chatbot_py import keyword_match
        result = keyword_match("I feel feverish and I am shivering")
        self.assertEqual(result, "fever_symptoms")

    def test_keyword_match_diabetes(self):
        from chatbot_py import keyword_match
        result = keyword_match("I have frequent urination and I'm always thirsty")
        self.assertEqual(result, "diabetes_symptoms")

    def test_keyword_match_headache(self):
        from chatbot_py import keyword_match
        result = keyword_match("I have a terrible migraine")
        self.assertEqual(result, "headache_symptoms")

    def test_keyword_match_anxiety(self):
        from chatbot_py import keyword_match
        result = keyword_match("I feel anxious and have racing thoughts")
        self.assertEqual(result, "anxiety_symptoms")

    def test_keyword_match_none(self):
        from chatbot_py import keyword_match
        result = keyword_match("what is the weather today in paris")
        self.assertIsNone(result)

    def test_keyword_fallback_returns_string(self):
        from chatbot_py import keyword_fallback
        result = keyword_fallback("asdfghjkl random gibberish")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_synonym_normalization(self):
        from chatbot_py import keyword_match
        result = keyword_match("my bp is really high")
        self.assertEqual(result, "hypertension_symptoms")

    def test_keyword_match_multi_returns_sorted(self):
        from chatbot_py import keyword_match_multi
        results = keyword_match_multi("I have a fever and headache")
        self.assertIsInstance(results, list)
        if len(results) > 1:
            self.assertGreaterEqual(results[0][1], results[1][1])

    def test_multi_symptom_fallback(self):
        from chatbot_py import keyword_fallback
        result = keyword_fallback("I have a fever and also my head is pounding")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestChatIntegration(unittest.TestCase):
    """Integration tests for the main chat() function."""

    def test_chat_returns_string(self):
        from chatbot_py import chat
        result = chat("hello")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_chat_greeting(self):
        from chatbot_py import chat
        result = chat("hi there")
        self.assertIsInstance(result, str)

    def test_chat_fallback_on_gibberish(self):
        from chatbot_py import chat
        result = chat("asdfghjkl xyzzy blorp")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_chat_medical_query(self):
        from chatbot_py import chat
        result = chat("I have a runny nose and sore throat")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_chat_prevention_query(self):
        from chatbot_py import chat
        result = chat("what medicines can I take for fever")
        self.assertIsInstance(result, str)

    def test_context_followup(self):
        from chatbot_py import chat, _context
        chat("I have a headache and my head is pounding")
        result = chat("tell me more")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for robustness."""

    def test_empty_input(self):
        from chatbot_py import chat
        result = chat("")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_whitespace_only(self):
        from chatbot_py import chat
        result = chat("   ")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_very_long_input(self):
        from chatbot_py import chat
        long_msg = "I have a headache " * 100
        result = chat(long_msg)
        self.assertIsInstance(result, str)

    def test_special_characters(self):
        from chatbot_py import chat
        result = chat("!@#$%^&*()_+{}|:<>?")
        self.assertIsInstance(result, str)

    def test_unicode_input(self):
        from chatbot_py import chat
        result = chat("I have a headache \u2014 it\u2019s really bad")
        self.assertIsInstance(result, str)

    def test_numbers_only(self):
        from chatbot_py import chat
        result = chat("12345 67890")
        self.assertIsInstance(result, str)

    def test_single_character(self):
        from chatbot_py import chat
        result = chat("a")
        self.assertIsInstance(result, str)


class TestConfidenceRouting(unittest.TestCase):
    """Test the prediction and confidence routing logic."""

    def test_predict_class_returns_list(self):
        from chatbot_py import predict_class
        result = predict_class("hello")
        self.assertIsInstance(result, list)

    def test_predict_class_probability_format(self):
        from chatbot_py import predict_class
        result = predict_class("I have a cold")
        if result:
            self.assertIn('intent', result[0])
            self.assertIn('probability', result[0])
            prob = float(result[0]['probability'])
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_predict_class_normalized_tags(self):
        from chatbot_py import predict_class
        result = predict_class("I have diabetes symptoms")
        for pred in result:
            tag = pred['intent']
            self.assertEqual(tag, tag.lower(),
                             f"Tag '{tag}' is not lowercase")
            self.assertNotIn(' ', tag,
                             f"Tag '{tag}' contains spaces")

    def test_gibberish_low_confidence(self):
        from chatbot_py import predict_class
        result = predict_class("xyzzy blorp fleem")
        if result:
            top_prob = float(result[0]['probability'])
            self.assertLess(top_prob, 0.70)


class TestMultiTurnConversation(unittest.TestCase):
    """Test multi-turn conversation with context tracking."""

    def test_symptom_then_followup(self):
        from chatbot_py import chat, _context
        chat("I have a really bad headache")
        result = chat("tell me more")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_context_resets_on_new_topic(self):
        from chatbot_py import chat, _context
        chat("I have a cold")
        chat("I feel anxious and nervous")
        last = _context.get("last_intent")
        self.assertIsNotNone(last)

    def test_prevention_followup_contains_treatment(self):
        from chatbot_py import chat, _context
        # Force context to a known symptom
        _context["last_intent"] = "fever_symptoms"
        result = chat("tell me more")
        # Should provide prevention/treatment info
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 20)  # should be substantial


class TestResponseContent(unittest.TestCase):
    """Test response quality and variety."""

    def test_fallback_response_is_helpful(self):
        from chatbot_py import get_fallback_response
        for _ in range(10):
            response = get_fallback_response()
            has_helpful_word = any(w in response.lower() for w in
                                  ["symptom", "health", "rephrase", "describe", "feeling"])
            self.assertTrue(has_helpful_word,
                            f"Fallback '{response}' doesn't guide user")

    def test_response_variety(self):
        from chatbot_py import chat
        responses = set()
        for _ in range(10):
            responses.add(chat("hello"))
        self.assertGreater(len(responses), 1,
                           "Repeated greetings should produce varied responses")

    def test_all_intents_have_responses(self):
        with open('intents.json') as f:
            data = json.load(f)
        for intent in data['intents']:
            self.assertGreater(len(intent['responses']), 0,
                               f"Intent '{intent['tag']}' has no responses")

    def test_all_intents_have_multiple_responses(self):
        with open('intents.json') as f:
            data = json.load(f)
        for intent in data['intents']:
            if intent['tag'] != 'noanswer':
                self.assertGreaterEqual(len(intent['responses']), 2,
                                        f"Intent '{intent['tag']}' has fewer than 2 responses")


class TestTagNormalization(unittest.TestCase):
    """Verify all intent tags follow the normalized format."""

    def test_all_tags_lowercase_no_spaces(self):
        with open('intents.json') as f:
            data = json.load(f)
        for intent in data['intents']:
            tag = intent['tag']
            self.assertEqual(tag, tag.lower(),
                             f"Tag '{tag}' is not lowercase")
            self.assertNotIn(' ', tag,
                             f"Tag '{tag}' contains spaces")

    def test_keyword_map_tags_match_intents(self):
        from chatbot_py import KEYWORD_MAP
        with open('intents.json') as f:
            data = json.load(f)
        intent_tags = {i['tag'] for i in data['intents']}
        for tag in KEYWORD_MAP:
            self.assertIn(tag, intent_tags,
                          f"KEYWORD_MAP tag '{tag}' not found in intents.json")

    def test_classes_pkl_matches_intents(self):
        import pickle
        with open('intents.json') as f:
            data = json.load(f)
        classes = pickle.load(open('classes.pkl', 'rb'))
        intent_tags = sorted({i['tag'] for i in data['intents']})
        self.assertEqual(sorted(classes), intent_tags)

    def test_symptom_prevention_correspondence(self):
        """Every symptom intent should have a matching prevention intent."""
        with open('intents.json') as f:
            data = json.load(f)
        tags = {i['tag'] for i in data['intents']}
        for tag in tags:
            if tag.endswith("_symptoms"):
                prevention = tag.replace("_symptoms", "_prevention")
                self.assertIn(prevention, tags,
                              f"Symptom intent '{tag}' has no corresponding '{prevention}'")


class TestSafetyResponses(unittest.TestCase):
    """Test emergency/safety handling for critical symptoms."""

    def test_suicidal_input_triggers_emergency(self):
        from chatbot_py import chat
        result = chat("I want to kill myself")
        self.assertIn("emergency", result.lower())

    def test_heart_attack_triggers_emergency(self):
        from chatbot_py import chat
        result = chat("I think I am having a heart attack")
        self.assertIn("emergency", result.lower())

    def test_overdose_triggers_emergency(self):
        from chatbot_py import chat
        result = chat("I took too many pills and overdosed")
        self.assertIn("911", result)

    def test_cant_breathe_emergency(self):
        from chatbot_py import chat
        result = chat("my child cant breathe at all and is turning blue")
        self.assertIn("911", result)

    def test_safety_check_returns_none_for_normal(self):
        from chatbot_py import check_emergency
        result = check_emergency("I have a headache")
        self.assertIsNone(result)

    def test_safety_overrides_other_routing(self):
        """Emergency check should happen before model prediction."""
        from chatbot_py import chat
        # This has both depression keywords and emergency keywords
        result = chat("I feel depressed and I want to end my life")
        self.assertIn("emergency", result.lower())


class TestConfidenceMargin(unittest.TestCase):
    """Test confidence margin-based routing to avoid overconfident wrong answers."""

    def test_is_confident_requires_margin(self):
        from chatbot_py import _is_confident
        # Top prediction high but second too close -> not confident
        ints = [
            {'intent': 'fever_symptoms', 'probability': '0.80'},
            {'intent': 'common_cold_symptoms', 'probability': '0.75'},
        ]
        self.assertFalse(_is_confident(ints))

    def test_is_confident_with_good_margin(self):
        from chatbot_py import _is_confident
        ints = [
            {'intent': 'fever_symptoms', 'probability': '0.85'},
            {'intent': 'common_cold_symptoms', 'probability': '0.05'},
        ]
        self.assertTrue(_is_confident(ints))

    def test_is_confident_single_prediction(self):
        from chatbot_py import _is_confident
        ints = [{'intent': 'greetings', 'probability': '0.90'}]
        self.assertTrue(_is_confident(ints))

    def test_is_confident_low_prob(self):
        from chatbot_py import _is_confident
        ints = [{'intent': 'greetings', 'probability': '0.40'}]
        self.assertFalse(_is_confident(ints))

    def test_is_confident_empty(self):
        from chatbot_py import _is_confident
        self.assertFalse(_is_confident([]))


class TestModelComparison(unittest.TestCase):
    """Validate training metrics and model comparison results."""

    def test_training_metrics_exist(self):
        with open('training_metrics.json') as f:
            metrics = json.load(f)
        self.assertIn('best_model', metrics)
        self.assertIn('comparison', metrics)

    def test_all_models_compared(self):
        with open('training_metrics.json') as f:
            metrics = json.load(f)
        self.assertIn('logistic_regression', metrics['comparison'])
        self.assertIn('svm', metrics['comparison'])
        self.assertIn('neural_network', metrics['comparison'])

    def test_best_model_has_highest_val_accuracy(self):
        with open('training_metrics.json') as f:
            metrics = json.load(f)
        best = metrics['best_model']
        best_val = metrics['comparison'][best]['val_accuracy']
        for name, result in metrics['comparison'].items():
            self.assertGreaterEqual(best_val, result['val_accuracy'],
                                    f"{best} (val={best_val}) should beat {name} (val={result['val_accuracy']})")


class TestIntentAccuracy(unittest.TestCase):
    """Test that the model correctly classifies training patterns."""

    def test_intent_classification_accuracy(self):
        from chatbot_py import predict_class

        with open('intents.json') as f:
            intents_data = json.load(f)

        correct = 0
        total = 0
        mismatches = []

        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag in ('noanswer', 'followup'):
                continue
            for pattern in intent['patterns']:
                total += 1
                predictions = predict_class(pattern)
                if predictions and predictions[0]['intent'] == tag:
                    correct += 1
                else:
                    predicted = predictions[0]['intent'] if predictions else "NONE"
                    mismatches.append(f"  '{pattern}' -> expected '{tag}', got '{predicted}'")

        accuracy = correct / total if total > 0 else 0
        print(f"\nTraining set accuracy: {correct}/{total} = {accuracy:.1%}")
        if mismatches:
            print(f"Mismatches ({len(mismatches)}):")
            for m in mismatches[:10]:
                print(m)

        self.assertGreater(accuracy, 0.50,
                           f"Intent accuracy too low: {accuracy:.1%}")


class TestRealWorldEvaluation(unittest.TestCase):
    """Evaluate on held-out realistic queries NOT in training data."""

    def test_eval_queries(self):
        from chatbot_py import chat, predict_class, keyword_match, check_emergency

        with open('eval_queries.json') as f:
            eval_data = json.load(f)

        correct = 0
        total = 0
        mismatches = []

        for q in eval_data['queries']:
            total += 1
            user_input = q['input']
            expected = q['expected']

            # Determine what the system would classify as
            # (replicate chat() logic to get the chosen intent)
            if expected == "emergency":
                # For emergency queries, check if safety check catches them
                if check_emergency(user_input):
                    correct += 1
                    continue
                else:
                    mismatches.append(f"  '{user_input}' -> expected 'emergency', safety check missed")
                    continue

            preds = predict_class(user_input)
            kw = keyword_match(user_input)

            # The system uses the top prediction or keyword fallback
            if preds:
                predicted = preds[0]['intent']
            elif kw:
                predicted = kw
            else:
                predicted = "noanswer"

            # Also count keyword match as correct
            if predicted == expected or kw == expected:
                correct += 1
            else:
                mismatches.append(f"  '{user_input}' -> expected '{expected}', got pred='{predicted}' kw='{kw}'")

        accuracy = correct / total if total > 0 else 0
        print(f"\nReal-world eval accuracy: {correct}/{total} = {accuracy:.1%}")
        if mismatches:
            print(f"Mismatches ({len(mismatches)}):")
            for m in mismatches:
                print(m)

        # We want at least 65% on real-world queries
        # (hybrid system with keyword fallback should achieve this)
        self.assertGreater(accuracy, 0.65,
                           f"Real-world eval accuracy too low: {accuracy:.1%}")


if __name__ == '__main__':
    unittest.main()
