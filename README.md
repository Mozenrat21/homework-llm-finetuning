# Lesson 17 — LLM Fine-Tuning in Production

## 1. Опис завдання

Мета домашнього завдання — перевірити, чи може fine-tuned Llama 3.1 8B краще розв'язувати вузьку бізнес-задачу, ніж базова 8B модель без fine-tuning.

Бізнес-контекст: SaaS-компанія отримує приблизно 50 000 support emails на день. CTO хоче автоматично перетворювати кожен email у JSON для CRM.

Потрібно витягувати такі поля:

```json
{
  "customer_name": "string or null",
  "product": "string",
  "issue_category": "billing | technical | account | feature_request | other",
  "urgency": "low | medium | high | critical",
  "summary": "short sentence"
}
```

Гіпотеза: fine-tuned Llama 3.1 8B на цій конкретній задачі може перевершити або суттєво наблизитися до більшої 70B моделі, але з дешевшим inference.

---

## 2. Використані дані

Для роботи використано матеріали з репозиторію курсу:

* `data/eval.jsonl` — eval set;
* `data/train.jsonl` — training set;
* `scripts/evaluate.py`, `scripts/evaluate_together.py` — референсні evaluation-скрипти з матеріалів курсу;
* `scripts/generate_data.py`, `scripts/train_upload.py`, `scripts/poll_ft.py` — додаткові скрипти з матеріалів курсу.

Основне навчання виконувалося в Google Colab notebook:

```text
notebooks/lesson17_llama31_8b_baseline_finetune_clean.ipynb
```

---

## 3. Перевірка даних

Перед baseline і fine-tuning було виконано перевірку dataset через власний скрипт:

```text
scripts/check_data.py
```

Результат перевірки збережено у файлі:

```text
results/data_check.json
```

Підсумок перевірки:

| Перевірка                | Результат |
| ------------------------ | --------: |
| Eval examples            |        30 |
| Train examples           |       300 |
| Eval schema errors       |         0 |
| Train schema errors      |         0 |
| Email overlap train/eval |         0 |
| Duplicate eval emails    |         0 |
| Duplicate train emails   |         0 |

Розподіл `urgency` в eval set:

| Urgency  | Кількість |
| -------- | --------: |
| low      |        10 |
| medium   |        11 |
| high     |         6 |
| critical |         3 |

Розподіл `urgency` в train set:

| Urgency  | Кількість |
| -------- | --------: |
| low      |       108 |
| medium   |       132 |
| high     |        38 |
| critical |        22 |

Висновок: dataset валідний, train/eval overlap відсутній, можна чесно порівнювати baseline і fine-tuned модель на одному eval set.

---

## 4. Baseline evaluation

Baseline-модель:

```text
unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
```

Baseline оцінювався без task-specific fine-tuning на тих самих 30 eval examples.

Файл з результатами:

```text
results/baseline_8b_metrics.json
```

Baseline results:

| Метрика                 | Значення |
| ----------------------- | -------: |
| JSON valid              |    96.7% |
| Exact match             |    30.0% |
| customer_name accuracy  |    96.7% |
| product accuracy        |    80.0% |
| issue_category accuracy |    76.7% |
| urgency accuracy        |    53.3% |
| summary accuracy        |    50.0% |
| Avg input tokens        |     24.1 |
| Avg output tokens       |     51.0 |
| Avg seconds/example     |     6.56 |

Baseline майже завжди повертав валідний JSON, але мав слабку якість по `urgency` і `summary`. Для бізнес-задачі це критично, тому що неправильна `urgency` може призвести до пропуску critical incidents.

---

## 5. Fine-tuning

Fine-tuning виконано в Google Colab на T4 GPU.

Параметри:

| Параметр             | Значення                                      |
| -------------------- | --------------------------------------------- |
| Base model           | Llama 3.1 8B Instruct                         |
| Model checkpoint     | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| Method               | QLoRA                                         |
| Quantization         | 4-bit                                         |
| LoRA r               | 16                                            |
| LoRA alpha           | 32                                            |
| Epochs               | 3                                             |
| Train examples       | 300                                           |
| Trainable parameters | 41,943,040                                    |
| Total parameters     | 8,072,204,288                                 |
| Trainable share      | 0.52%                                         |
| Training time        | 7.2 minutes                                   |
| Final train loss     | 0.2452                                        |

Training summary:

```text
results/training_summary.json
```

LoRA adapter:

```text
LoRA adapter було збережено локально після тренування, але не додано в GitHub через обмеження розміру файлів. У репозиторії залишені notebook, training summary та evaluation metrics, яких достатньо для перевірки й відтворення експерименту.
```

---

## 6. Fine-tuned evaluation

Fine-tuned модель оцінювалася на тому самому eval set з 30 прикладів.

Файл з результатами:

```text
results/finetuned_8b_metrics.json
```

Fine-tuned results:

| Метрика                 | Значення |
| ----------------------- | -------: |
| JSON valid              |   100.0% |
| Exact match             |    70.0% |
| customer_name accuracy  |    93.3% |
| product accuracy        |    96.7% |
| issue_category accuracy |    90.0% |
| urgency accuracy        |    93.3% |
| summary accuracy        |    83.3% |
| Avg input tokens        |     24.1 |
| Avg output tokens       |     42.0 |
| Avg seconds/example     |     3.10 |

---

## 7. Порівняння baseline vs fine-tuned

| Метрика                 | Baseline 8B | Fine-tuned 8B |      Зміна |
| ----------------------- | ----------: | ------------: | ---------: |
| JSON valid              |       96.7% |        100.0% |  +3.3 п.п. |
| Exact match             |       30.0% |         70.0% | +40.0 п.п. |
| customer_name accuracy  |       96.7% |         93.3% |  -3.4 п.п. |
| product accuracy        |       80.0% |         96.7% | +16.7 п.п. |
| issue_category accuracy |       76.7% |         90.0% | +13.3 п.п. |
| urgency accuracy        |       53.3% |         93.3% | +40.0 п.п. |
| summary accuracy        |       50.0% |         83.3% | +33.3 п.п. |
| Avg output tokens       |        51.0 |          42.0 |       -9.0 |
| Avg seconds/example     |        6.56 |          3.10 |     швидше |

Fine-tuning суттєво покращив якість саме на ключових полях:

* `urgency`: 53.3% → 93.3%;
* `summary`: 50.0% → 83.3%;
* `exact_match`: 30.0% → 70.0%.

---

## 8. Cost & breakeven

Training cost у цьому експерименті:

```text
$0
```

Причина: навчання виконувалося в Google Colab free на T4 GPU.

Training time:

```text
7.2 minutes
```

Для бізнесу з 50 000 emails/day перевага власної fine-tuned 8B моделі потенційно з'являється тоді, коли:

1. traffic стабільно великий;
2. latency власної моделі прийнятна;
3. є інфраструктура для serving;
4. вартість inference API на великій моделі перевищує витрати на self-hosted inference;
5. якість fine-tuned 8B достатня для production або human-in-the-loop сценарію.

На T4 у Colab fine-tuned модель показала приблизно 3.10 сек/приклад на eval. Для 50 000 emails/day це означає, що один T4 у такому режимі може бути недостатнім для production, якщо потрібна швидка обробка всього потоку. Для production треба тестувати inference на більш оптимізованому serving stack: vLLM, TGI або інший inference server.

---

## 9. Що вийшло добре

Fine-tuning підтвердив гіпотезу, що для вузької structured extraction задачі маленька 8B модель після донавчання може суттєво покращити якість.

Найбільші покращення:

* значно краща класифікація `urgency`;
* краща класифікація `issue_category`;
* коротші outputs;
* вищий exact match;
* стабільний валідний JSON.

Особливо важливо, що `urgency` виросла до 93.3%, бо саме це поле впливає на escalation і routing critical incidents.

---

## 10. Що не вийшло / обмеження

1. Eval set містить лише 30 прикладів. Цього достатньо для навчального експерименту, але мало для production-висновків.
2. Training set містить лише 300 прикладів. Є ризик overfitting.
3. Fine-tuned модель тестувалася на синтетичному/навчальному наборі, а не на реальних production emails.
4. Не було окремого test set після fine-tuning.
5. Не проводилося порівняння з реальною Llama 70B у цьому ж середовищі.
6. Не перевірялися adversarial cases, multilingual emails, довгі email threads, attachments, HTML emails.
7. Serving latency у Colab не дорівнює production latency.
8. Є ризик train/serve skew, якщо production prompt або формат email буде відрізнятися від training data.

---

## 11. Бізнес-рекомендація

Fine-tuning Llama 3.1 8B для задачі support email → CRM JSON виглядає перспективним. У цьому експерименті exact match зріс з 30.0% до 70.0%, а accuracy для `urgency` — з 53.3% до 93.3%.

Рекомендація: не запускати модель одразу в повністю автоматичний production без контролю. Краще почати з human-in-the-loop сценарію:

1. модель автоматично заповнює JSON;
2. critical/high tickets додатково перевіряються;
3. логуються prediction, confidence proxy, raw email і виправлення оператора;
4. на основі реальних виправлень збирається наступний training set;
5. після розширення eval set і production-like тестів можна приймати рішення про повну автоматизацію.

Для 50 000 emails/day fine-tuned 8B може бути економічно цікавішою за API до великої моделі, але тільки після окремого benchmark inference вартості, latency і throughput на production hardware.

---

## 12. Як відтворити

### 1. Перевірити дані локально

```bash
python scripts/check_data.py
```

Очікуваний результат:

```text
STATUS: PASS
```

### 2. Відкрити notebook

```text
notebooks/lesson17_llama31_8b_baseline_finetune_clean.ipynb
```

### 3. Запустити в Google Colab

Runtime:

```text
T4 GPU
```

Notebook виконує:

1. встановлення залежностей;
2. завантаження train/eval даних;
3. baseline evaluation;
4. QLoRA fine-tuning;
5. fine-tuned evaluation;
6. збереження метрик і adapter.

---

## 13. Файли результатів

| Файл                                                          | Призначення                    |
| ------------------------------------------------------------- | ------------------------------ |
| `results/data_check.json`                                     | перевірка dataset              |
| `results/baseline_8b_metrics.json`                            | baseline metrics               |
| `results/training_summary.json`                               | параметри і результат training |
| `results/finetuned_8b_metrics.json`                           | fine-tuned metrics             |
| `results/llama31_8b_lora_adapter.zip`                         | LoRA adapter                   |
| `notebooks/lesson17_llama31_8b_baseline_finetune_clean.ipynb` | основний notebook експерименту |

---

## 14. Висновок

Гіпотеза підтвердилася для навчального eval set: fine-tuned Llama 3.1 8B суттєво перевершила baseline 8B на задачі structured JSON extraction з support emails.

Найважливіший результат:

```text
Exact match: 30.0% → 70.0%
Urgency accuracy: 53.3% → 93.3%
```

Це показує, що fine-tuning малої моделі може бути практичним рішенням для вузьких бізнес-задач, де потрібен стабільний формат відповіді, контроль вартості inference і краща якість на доменних прикладах.
