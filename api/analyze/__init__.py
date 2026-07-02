import json
import logging
import os
import time
import urllib.error
import urllib.request

import azure.functions as func

# Read these from Azure environment variables / application settings —
# never hard-code the key here.
ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20
MAX_CHARS = 5000


def main(req: func.HttpRequest) -> func.HttpResponse:
    global ENDPOINT, KEY
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    if not endpoint or not key:
        return _json_response(
            {"error": "Server is missing LANGUAGE_ENDPOINT / LANGUAGE_KEY environment variables."},
            500,
        )
    ENDPOINT = endpoint
    KEY = key

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    text = (body.get("text") or "").strip()
    if not text:
        return _json_response({"error": 'Request body must include non-empty "text".'}, 400)
    if len(text) > MAX_CHARS:
        return _json_response({"error": f"Text must be {MAX_CHARS} characters or fewer."}, 400)

    # 1. Detect language
    try:
        lang_res = _call_language("LanguageDetection", text)
        lang_doc = lang_res["results"]["documents"][0]
        detected_lang = lang_doc["detectedLanguage"]["iso6391Name"]
        detected_lang_name = lang_doc["detectedLanguage"]["name"]
        detected_lang_score = lang_doc["detectedLanguage"]["confidenceScore"]
    except Exception:
        logging.exception("Language detection failed, falling back to 'en'")
        detected_lang = "en"
        detected_lang_name = "English"
        detected_lang_score = 0.0

    # 2. Sentiment analysis
    try:
        sentiment_doc = _call_language("SentimentAnalysis", text, language=detected_lang)["results"]["documents"][0]
        sentiment = sentiment_doc["sentiment"]
        confidence_scores = sentiment_doc["confidenceScores"]
    except Exception:
        logging.exception("Sentiment analysis failed")
        sentiment = "neutral"
        confidence_scores = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    # 3. Key phrase extraction
    try:
        keyphrase_doc = _call_language("KeyPhraseExtraction", text, language=detected_lang)["results"]["documents"][0]
        key_phrases = keyphrase_doc.get("keyPhrases", [])
    except Exception:
        logging.exception("Key phrase extraction failed")
        key_phrases = []

    # 4. Entity recognition
    try:
        entity_doc = _call_language("EntityRecognition", text, language=detected_lang)["results"]["documents"][0]
        entities = [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ]
    except Exception:
        logging.exception("Entity recognition failed")
        entities = []

    # 5. PII Entity recognition
    try:
        pii_res = _call_language("PiiEntityRecognition", text, language=detected_lang)
        pii_doc = pii_res["results"]["documents"][0]
        redacted_text = pii_doc.get("redactedText", text)
        pii_entities = [
            {"text": e["text"], "category": e["category"]}
            for e in pii_doc.get("entities", [])
        ]
    except Exception:
        logging.exception("PII detection failed")
        redacted_text = text
        pii_entities = []

    # 6. Extractive Summarization (asynchronous job pattern with sync backend polling)
    try:
        summary_sentences = _call_summarization_job(text, language=detected_lang)
    except Exception:
        logging.exception("Extractive summarization failed")
        summary_sentences = []

    result = {
        "sentiment": sentiment,
        "confidenceScores": confidence_scores,
        "keyPhrases": key_phrases,
        "entities": entities,
        "language": {
            "name": detected_lang_name,
            "iso6391Name": detected_lang,
            "confidenceScore": detected_lang_score,
        },
        "pii": {
            "redactedText": redacted_text,
            "entities": pii_entities,
        },
        "summary": summary_sentences,
    }
    return _json_response(result, 200)


def _call_language(kind: str, text: str, language: str = None) -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"
    doc = {"id": "1", "text": text}
    if language:
        doc["language"] = language
    payload = {
        "kind": kind,
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {"documents": [doc]},
    }
    if kind == "PiiEntityRecognition":
        payload["parameters"]["domain"] = "none"

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{kind} failed: {exc.code} {detail}") from exc


def _call_summarization_job(text: str, language: str) -> list:
    url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={API_VERSION}"
    payload = {
        "displayName": "Extractive Summarization Task",
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "language": language,
                    "text": text
                }
            ]
        },
        "tasks": [
            {
                "kind": "ExtractiveSummarization",
                "taskName": "Extractive Summarization Task 1",
                "parameters": {
                    "sentenceCount": 3,
                    "sortBy": "Offset"
                }
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            operation_location = resp.info().get("operation-location")
            if not operation_location:
                logging.error("No operation-location header found in summarization response")
                return []
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        logging.error(f"Failed to submit summarization job: {exc.code} {detail}")
        return []
    except Exception as exc:
        logging.error(f"Failed to submit summarization job: {exc}")
        return []

    # Polling loop
    poll_request = urllib.request.Request(
        operation_location,
        headers={
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="GET",
    )

    max_polls = 10
    poll_interval = 1.0
    for attempt in range(max_polls):
        time.sleep(poll_interval)
        try:
            with urllib.request.urlopen(poll_request, timeout=TIMEOUT_SECONDS) as resp:
                job_res = json.loads(resp.read().decode("utf-8"))
                status = job_res.get("status")
                if status == "succeeded":
                    sentences = []
                    for item in job_res.get("tasks", {}).get("items", []):
                        if item.get("kind") == "ExtractiveSummarization" and item.get("status") == "succeeded":
                            docs = item.get("results", {}).get("documents", [])
                            if docs:
                                sentences = [
                                    {"text": s["text"], "rankScore": s.get("rankScore", 0)}
                                    for s in docs[0].get("sentences", [])
                                ]
                    return sentences
                elif status == "failed":
                    logging.error(f"Summarization job failed in status: {job_res}")
                    return []
        except Exception as exc:
            logging.warning(f"Error polling summarization job on attempt {attempt+1}: {exc}")

    logging.warning("Summarization job timed out")
    return []


def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )
