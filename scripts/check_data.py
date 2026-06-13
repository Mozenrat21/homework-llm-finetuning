import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_FILE = DATA_DIR / "eval.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
OUTPUT_FILE = RESULTS_DIR / "data_check.json"

FIELDS = ["customer_name", "product", "issue_category", "urgency", "summary"]
ALLOWED_CATEGORIES = {"billing", "technical", "account", "feature_request", "other"}
ALLOWED_URGENCIES = {"low", "medium", "high", "critical"}


def load_jsonl(file_path: Path) -> list[dict]:
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{file_path} line {line_number}: invalid JSON: {error}") from error

    return records


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def text_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_expected_object(obj: dict, source: str) -> list[str]:
    errors = []

    if not isinstance(obj, dict):
        return [f"{source}: expected object is not a dict"]

    missing_fields = [field for field in FIELDS if field not in obj]
    extra_fields = [field for field in obj.keys() if field not in FIELDS]

    if missing_fields:
        errors.append(f"{source}: missing fields: {missing_fields}")

    if extra_fields:
        errors.append(f"{source}: extra fields: {extra_fields}")

    if "issue_category" in obj and obj["issue_category"] not in ALLOWED_CATEGORIES:
        errors.append(f"{source}: invalid issue_category: {obj['issue_category']}")

    if "urgency" in obj and obj["urgency"] not in ALLOWED_URGENCIES:
        errors.append(f"{source}: invalid urgency: {obj['urgency']}")

    if "summary" in obj and not isinstance(obj["summary"], str):
        errors.append(f"{source}: summary must be string")

    if "product" in obj and not isinstance(obj["product"], str):
        errors.append(f"{source}: product must be string")

    if "customer_name" in obj and obj["customer_name"] is not None and not isinstance(obj["customer_name"], str):
        errors.append(f"{source}: customer_name must be string or null")

    return errors


def extract_train_parts(record: dict, index: int) -> tuple[str | None, dict | None, list[str]]:
    errors = []
    messages = record.get("messages")

    if not isinstance(messages, list):
        return None, None, [f"train line {index}: messages must be a list"]

    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

    if len(user_messages) != 1:
        errors.append(f"train line {index}: expected exactly 1 user message, got {len(user_messages)}")

    if len(assistant_messages) != 1:
        errors.append(f"train line {index}: expected exactly 1 assistant message, got {len(assistant_messages)}")

    user_email = None
    assistant_json = None

    if user_messages:
        user_email = user_messages[0].get("content")
        if not isinstance(user_email, str) or not user_email.strip():
            errors.append(f"train line {index}: user content must be non-empty string")

    if assistant_messages:
        assistant_content = assistant_messages[0].get("content")

        try:
            assistant_json = json.loads(assistant_content)
        except Exception as error:
            errors.append(f"train line {index}: assistant content is not valid JSON: {error}")

    return user_email, assistant_json, errors


def main() -> None:
    eval_records = load_jsonl(EVAL_FILE)
    train_records = load_jsonl(TRAIN_FILE)

    eval_errors = []
    train_errors = []

    eval_emails = []
    eval_categories = []
    eval_urgencies = []

    for index, record in enumerate(eval_records, start=1):
        email = record.get("email")
        expected = record.get("expected")

        if not isinstance(email, str) or not email.strip():
            eval_errors.append(f"eval line {index}: email must be non-empty string")
        else:
            eval_emails.append(email)

        eval_errors.extend(validate_expected_object(expected, f"eval line {index} expected"))

        if isinstance(expected, dict):
            eval_categories.append(expected.get("issue_category"))
            eval_urgencies.append(expected.get("urgency"))

    train_emails = []
    train_categories = []
    train_urgencies = []

    for index, record in enumerate(train_records, start=1):
        user_email, assistant_json, errors = extract_train_parts(record, index)
        train_errors.extend(errors)

        if user_email:
            train_emails.append(user_email)

        if assistant_json:
            train_errors.extend(validate_expected_object(assistant_json, f"train line {index} assistant"))
            train_categories.append(assistant_json.get("issue_category"))
            train_urgencies.append(assistant_json.get("urgency"))

    eval_email_hashes = {text_hash(email) for email in eval_emails}
    train_email_hashes = {text_hash(email) for email in train_emails}
    overlap_hashes = eval_email_hashes.intersection(train_email_hashes)

    duplicate_eval_emails = {
        email: count
        for email, count in Counter(normalize_text(email) for email in eval_emails).items()
        if count > 1
    }

    duplicate_train_emails = {
        email: count
        for email, count in Counter(normalize_text(email) for email in train_emails).items()
        if count > 1
    }

    report = {
        "eval_count": len(eval_records),
        "train_count": len(train_records),
        "eval_errors_count": len(eval_errors),
        "train_errors_count": len(train_errors),
        "email_overlap_count": len(overlap_hashes),
        "duplicate_eval_emails_count": len(duplicate_eval_emails),
        "duplicate_train_emails_count": len(duplicate_train_emails),
        "eval_issue_category_distribution": dict(Counter(eval_categories)),
        "train_issue_category_distribution": dict(Counter(train_categories)),
        "eval_urgency_distribution": dict(Counter(eval_urgencies)),
        "train_urgency_distribution": dict(Counter(train_urgencies)),
        "eval_errors": eval_errors[:20],
        "train_errors": train_errors[:20],
        "passed": (
            len(eval_records) == 30
            and len(train_records) == 300
            and len(eval_errors) == 0
            and len(train_errors) == 0
            and len(overlap_hashes) == 0
        ),
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("DATA CHECK")
    print("=" * 60)
    print(f"Eval examples: {report['eval_count']}")
    print(f"Train examples: {report['train_count']}")
    print(f"Eval schema errors: {report['eval_errors_count']}")
    print(f"Train schema errors: {report['train_errors_count']}")
    print(f"Email overlap train/eval: {report['email_overlap_count']}")
    print(f"Duplicate eval emails: {report['duplicate_eval_emails_count']}")
    print(f"Duplicate train emails: {report['duplicate_train_emails_count']}")
    print()
    print("Eval urgency distribution:")
    print(report["eval_urgency_distribution"])
    print()
    print("Train urgency distribution:")
    print(report["train_urgency_distribution"])
    print()
    print(f"Saved report: {OUTPUT_FILE}")

    if report["passed"]:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        print("Open results/data_check.json for details")


if __name__ == "__main__":
    main()
