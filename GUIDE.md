# Complete Guide — Legal Document Summarization & Evaluation

> This guide explains everything from scratch. No assumptions about your background.  
> The original README is one page. This file is not.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Core Problem Being Solved](#2-the-core-problem-being-solved)
3. [Key Idea — Intent-Based Evaluation](#3-key-idea--intent-based-evaluation)
4. [How the Dataset Was Built](#4-how-the-dataset-was-built)
5. [Repository Structure — Every Folder Explained](#5-repository-structure--every-folder-explained)
6. [The Dataset in Detail](#6-the-dataset-in-detail)
7. [The Summarization Models](#7-the-summarization-models)
8. [How Evaluation Works](#8-how-evaluation-works)
9. [JointBERT — The Intent Extractor](#9-jointbert--the-intent-extractor)
10. [The Demo App](#10-the-demo-app)
11. [Running Everything on Google Colab](#11-running-everything-on-google-colab)
12. [Glossary](#12-glossary)

---

## 1. What Is This Project?

This is the code and data for a research paper published at **LREC 2022** (a top NLP conference):

> **"An Evaluation Framework for Legal Document Summarization"**  
> Mullick, Nandy, Kapadnis, Patnaik, Raghav, Kar — 2022  
> Paper: https://arxiv.org/abs/2205.08478

**In plain English:** A lawyer reading a 50-page court judgment needs to know quickly — what type of case is this? What happened? What were the key facts? The paper builds a system that:

1. Automatically **summarizes** long legal documents using 5 different methods
2. **Evaluates** how good those summaries are — not just by matching words, but by checking whether the summary captures the *intent* of the case (e.g., was it really about robbery? does the summary reflect that?)

---

## 2. The Core Problem Being Solved

### Why existing evaluation metrics are not enough

When researchers build a summarization system, they need a way to measure "is this summary good?". The standard tools are:

- **ROUGE** — counts how many words/phrases overlap between the generated summary and a human-written reference summary
- **BLEU** — similar word-overlap metric originally from machine translation
- **BERTScore** — uses BERT embeddings to compare meaning, not just words

**The problem:** These metrics only check if the summary *looks* similar to a reference. They don't check if the summary is actually *useful* for the specific task. For legal documents, what matters is whether the summary correctly represents the *type* and *intent* of the case.

**Example:** A Murder case summary that is 90% accurate in word overlap but fails to mention the murder weapon, the accused, or the conviction is a bad summary — but ROUGE would score it high.

### What this paper does differently

The paper introduces an **intent-based metric**. It asks:

> "Does the summary contain phrases that match the intent of the case category?"

A "Murder" case has intent phrases like *"convicted him to life imprisonment"*, *"committed the murder"*, *"stabbed the victim"*. A good summary should contain at least some of these. The metric checks this automatically and correlates better with what human evaluators actually prefer.

---

## 3. Key Idea — Intent-Based Evaluation

### What is "intent" here?

In this paper, "intent" means the specific phrases in a legal document that indicate what kind of case it is and what happened. These are not random sentences — they are the phrases that a lawyer would highlight as most important.

**Example from the dataset** (`ind_phrases_2/1_2.txt`):
```
committing the murder     0  23  7  3
assaulted with the help of knife   0  24  3  6
offence under section 307   0  26  4  4
section 302 of IPC   0  28  8  4
convicted him to life imprisonment  0  30  10  5
```

Each line is:
- **Column 1:** The intent phrase (what a human annotator marked as important)
- **Column 2:** Paragraph index where it appears
- **Column 3:** Sentence index within that paragraph
- **Column 4:** Word offset in the sentence
- **Column 5:** Length of the phrase in words

### How the metric works (simplified)

1. Take a generated summary
2. Extract intent phrases from it using the trained JointBERT model
3. Compare those extracted phrases against the gold-standard annotated intent phrases
4. Score = how well they match

This gives a score that correlates better with human judgment than ROUGE does.

---

## 4. How the Dataset Was Built

### Indian Dataset (93 documents)

- **Source:** Supreme Court of India judgments, downloaded from public legal databases
- **Categories:** Murder, Robbery, Land Dispute, Corruption (and mixed categories like "Murder and Robbery")
- **Manual annotation:** Human annotators (law students/practitioners) read each document and highlighted the intent phrases — the key sentences that define what the case is about
- **93 documents total**, each a real court judgment ranging from ~600 to ~16,000 words

### Australian Dataset (used for transfer learning experiments)

- **Source:** Australian court judgments
- **Categories:** LandDispute, Murder, Corruption, Robbery
- **Annotation:** Done automatically using the JointBERT model trained on Indian data (transfer learning — train on India, run on Australia)
- Used to test whether the intent extraction generalizes across jurisdictions

### Why only 93 documents?

Manual legal annotation is expensive and requires domain expertise. The annotators had to read full judgments and mark intent phrases — this takes hours per document. 93 documents is a realistic dataset size for this kind of specialized annotation work.

---

## 5. Repository Structure — Every Folder Explained

```
legaldocsvalution/
│
├── dataset/                        ← All the data
│   ├── indian_data/
│   │   ├── ind_text/               ← 93 raw court judgment text files (1.txt to 93.txt)
│   │   ├── ind_phrases_2/          ← 93 annotated intent phrase files (1_2.txt to 93_2.txt)
│   │   ├── ind_labels.csv          ← Category label for each document (Murder, Robbery, etc.)
│   │   └── ind_dataset_statistics.csv  ← Word/sentence/paragraph counts per document
│   │
│   └── australian_data/
│       ├── aus_text/               ← Raw Australian judgment text files
│       ├── aus_phrases/            ← Auto-extracted intent phrases (using JointBERT)
│       ├── aus_labels.csv          ← Category labels + annotated intent phrases
│       └── aus_data_statistics.csv ← Statistics per document
│
├── JointBert_input_dataset_indian/ ← Indian data reformatted for JointBERT training
│   ├── train/                      ← 1385 training examples
│   │   ├── label                   ← One intent class per line (Murder, Robbery, etc.)
│   │   ├── seq.in                  ← Input sentences (one per line)
│   │   └── seq.out                 ← BIO slot tags for each token
│   ├── dev/                        ← 146 validation examples
│   ├── test/                       ← 146 test examples
│   ├── intent_label.txt            ← List of intent classes (UNK, LandDispute, Murder, ...)
│   └── slot_label.txt              ← List of BIO slot tags (PAD, UNK, O, B-Murder, ...)
│
├── summarization_models/           ← 5 different summarization approaches
│   ├── Bert/
│   │   └── Extractive_summary_bert.ipynb   ← BERT-based extractive summarizer
│   ├── Graphical_Model/
│   │   ├── graphicalModel.py               ← Graph-based extractive summarizer
│   │   ├── crf_alltrain.model              ← Trained CRF model file
│   │   ├── data/                           ← Input text files (93 documents)
│   │   └── Graphical_Model.ipynb           ← Notebook to run the model
│   ├── LetSum Model/
│   │   ├── letsum.py                       ← LetSum extractive summarizer
│   │   └── data/                           ← Input text files
│   ├── Abstractive summarization/
│   │   └── abstractive_summarization.ipynb ← LSTM + Attention abstractive model
│   ├── Case_Summarizer.ipynb               ← CaseSummarizer (external tool)
│   ├── legal-led.ipynb                     ← Legal-LED transformer (GPU required)
│   └── requirements.txt                    ← Python dependencies for these models
│
├── demo_app_code/                  ← Streamlit web demo
│   ├── app.py                      ← Main app (paste a document, get summary + evaluation)
│   ├── graphicalModel.py           ← Copy of graphical model for app use
│   ├── letsum.py                   ← Copy of LetSum for app use
│   └── requirements.txt            ← App dependencies
│
├── LREC_poster.pdf                 ← Conference poster (visual summary of the paper)
├── LREC_poster_image.PNG           ← Poster as image
└── README.md                       ← Original minimal README
```

---

## 6. The Dataset in Detail

### Indian Data — `dataset/indian_data/`

#### Raw text files — `ind_text/1.txt` to `ind_text/93.txt`

Each file is a full court judgment. They start with a header line `"for educational use only"`, then the case name, then the full text. Example from document 10:

```
for educational use only
Balwan Singh vs The State Of Chhattisgarh on 6 August, 2019

Supreme Court of India
...
IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE JURISDICTION
...
[full judgment text continues for thousands of words]
```

#### Label file — `ind_labels.csv`

Tells you what category each document belongs to:

| file_number | old_class | new_class | duration |
|---|---|---|---|
| 1 | Murder | Murder | 2yrs and 6m |
| 11 | Robbery | Robbery | 4yrs and 3m |
| 21 | Land Dispute | Land Dispute | 13yrs and 1m |
| 41 | Murder and Robbery | Murder | 5yrs and 6m |

- **old_class:** The original multi-label (some cases involve multiple crime types)
- **new_class:** Simplified to one primary label for classification experiments
- **duration:** How long the case took from filing to judgment

**4 primary categories:**
- **Murder** — criminal homicide cases
- **Robbery** — theft with force/threat
- **Land Dispute** — property ownership/tenancy conflicts
- **Corruption** — bribery, abuse of public office (Prevention of Corruption Act)

#### Intent phrase files — `ind_phrases_2/1_2.txt` to `ind_phrases_2/93_2.txt`

The heart of the dataset. Each line is a manually annotated intent phrase:

```
committing the murder    0  23  7  3
assaulted with the help of knife   0  24  3  6
convicted him to life imprisonment  0  30  10  5
```

Format: `phrase  paragraph_idx  sentence_idx  word_offset  phrase_length`

These phrases were highlighted by human annotators as the most important intent-bearing phrases in each document. The model is trained to extract similar phrases from unseen documents.

#### Statistics file — `ind_dataset_statistics.csv`

```
file_name, word_count, sent_count, paragraph_count
61.txt,    1190,       54,         10
51.txt,    16287,      583,        30
```

Documents range from ~600 words (short judgments) to ~16,000 words (lengthy appeals). This shows why summarization is necessary — reading 16,000 words for every case is not practical.

### Australian Data — `dataset/australian_data/`

Same structure as Indian data. Key difference: the intent phrases in `aus_phrases/` were **not manually annotated** — they were generated automatically by running the JointBERT model (trained on Indian data) on the Australian documents.

The `aus_labels.csv` also includes the extracted phrases inline:

```
Doc Name,           Intent Phrase,                    Intent,       Sub Intent
07_1062_dispute,    refuses to take premium sports..., Land_Dispute, action
07_1062_dispute,    goes out of business,              Land_Dispute, description
```

There are sub-intent labels too: `action`, `description`, `claim` — a finer-grained categorization of what role the phrase plays in the case narrative.

---

## 7. The Summarization Models

The paper compares 5 different summarization approaches on the same 93 Indian documents. Here is what each one does and how to run it.

---

### Model 1 — BERT Extractive Summarizer

**Notebook:** `summarization_models/Bert/Extractive_summary_bert.ipynb`  
**Type:** Extractive (picks real sentences from the document, does not generate new text)  
**Runtime:** CPU, ~30 minutes for all 93 docs

**How it works:**
1. The document is split into sentences
2. BERT encodes each sentence into a vector (a list of numbers representing its meaning)
3. K-means clustering groups sentences that are semantically similar
4. The sentence closest to the center of each cluster is selected as a summary sentence
5. Output: a subset of the original sentences, up to 500 words

**What BERT is here:** `bert-base-uncased` from HuggingFace — a pre-trained transformer model that understands sentence meaning. The summarizer does NOT fine-tune BERT — it uses its embeddings directly.

**Library used:** `bert-extractive-summarizer` (wraps HuggingFace transformers)

**Output:** One `.txt` file per document in `/content/berts/`

---

### Model 2 — Graphical Model

**Notebook:** `summarization_models/Graphical_Model.ipynb`  
**Code:** `summarization_models/Graphical_Model/graphicalModel.py`  
**Type:** Extractive  
**Runtime:** CPU, ~40 minutes for all 93 docs × 3 fractions

**How it works:**
This is the most complex model in the paper. It combines two scoring methods:

**Step 1 — CRF (Conditional Random Field) scoring:**
- A CRF is a sequence labeling model (like a smarter version of a rule-based tagger)
- Pre-trained on legal text (`crf_alltrain.model`)
- Assigns an importance score to each sentence based on features like: position in document, sentence length, presence of legal keywords, section headers

**Step 2 — K-Mix Model scoring:**
- Scores sentences based on a mixture of: tf-idf (term frequency), position score, and similarity to document centroid
- K-mix = combining K different scoring signals with learned weights

**Step 3 — Combined ranking:**
- Sentences are ranked by combining both scores
- Top sentences by combined rank are selected until the target length is reached

**Output:** Summary files in `../../summary_outputs/Graphical_Model/summary_{fraction}/`  
(fraction = 0.3, 0.5, or 0.7 = percentage of original document length to keep)

---

### Model 3 — LetSum

**Code:** `summarization_models/LetSum Model/letsum.py`  
**Type:** Extractive  
**Origin:** A legal-domain summarization method from the literature

**How it works:**
LetSum is a rule-based extractive summarizer designed specifically for legal documents. It uses:
- Rhetorical role detection (identifying which sentences are "facts", "arguments", "holdings")
- Position-based scoring (sentences near section boundaries score higher)
- Legal keyword dictionaries

It is simpler than the Graphical Model but has the advantage of being interpretable — you can trace exactly why a sentence was selected.

---

### Model 4 — CaseSummarizer

**Notebook:** `summarization_models/Case_Summarizer.ipynb`  
**Source:** [Law-AI/summarization](https://github.com/Law-AI/summarization) (external public repo)  
**Type:** Extractive  
**Runtime:** Colab CPU, requires `python3`

**How it works:**
CaseSummarizer is a publicly available legal summarization tool. The notebook:
1. Clones the Law-AI summarization repo
2. Preprocesses documents (sentence splitting, cleaning)
3. Runs CaseSummarizer's scoring algorithm
4. Trims summaries to a target fraction of the original length (30%, 50%, or 70%)

The scoring combines: sentence position, named entity density, and similarity to the document's "theme" (centroid of all sentence TF-IDF vectors).

**Note:** This was originally Python 2. The notebook has been updated to run with Python 3.

---

### Model 5 — Legal-LED

**Notebook:** `summarization_models/legal-led.ipynb`  
**Type:** Abstractive (generates new text, not just extracted sentences)  
**Runtime:** GPU required (T4 on Colab), ~1-2 hours for 93 docs  
**Model:** `nsi319/legal-led-base-16384` from HuggingFace

**How it works:**
LED = Longformer Encoder-Decoder. This is a transformer model (like GPT or BERT) but designed for very long documents.

- **Standard BERT/GPT** can handle ~512 tokens (~400 words). A court judgment can be 10,000+ words — it would be truncated.
- **Longformer** uses "sliding window attention" — each token attends to nearby tokens + a few global tokens. This allows it to process up to **16,384 tokens** (~12,000 words).
- `nsi319/legal-led-base-16384` is a version of LED fine-tuned specifically on legal text.

**What it does:**
1. Tokenizes the document (up to 5,459 tokens — the pre-computed average length)
2. Feeds it through the LED encoder-decoder
3. Generates a summary using beam search (tries 2 candidate outputs, keeps the best)
4. Output is new text — the model writes its own sentences, not copies from the original

**Why GPU:**  
The model has ~160M parameters. Running inference on 93 documents at 5,459 tokens each on a CPU would take many hours. A T4 GPU does it in ~1-2 hours.

---

### Model 6 — Abstractive LSTM

**Notebook:** `summarization_models/Abstractive summarization/abstractive_summarization.ipynb`  
**Type:** Abstractive  
**Runtime:** Colab CPU/GPU, trains from scratch

**How it works:**
This is a classic sequence-to-sequence model with attention — the architecture that dominated NLP before transformers (2015-2019 era).

**Architecture:**
```
Input text → Embedding layer → 3-layer Bidirectional LSTM (Encoder)
                                          ↓
                              Attention mechanism (Bahdanau, 2015)
                                          ↓
                         Single LSTM (Decoder) → Dense softmax → Output word
```

- **Encoder:** Reads the input text and compresses it into hidden states
- **Attention:** At each decoding step, the decoder "looks back" at all encoder states and decides which parts of the input to focus on
- **Decoder:** Generates one word at a time, conditioned on what it has generated so far and the attended encoder context

**Training:**
- Trained on the 93 documents + their phrase summaries
- Uses early stopping on validation loss (stops when val loss stops improving)
- Optimizer: RMSprop
- Loss: sparse categorical crossentropy

**Important caveat:** 93 documents is a very small training set for an abstractive model. This is essentially a proof-of-concept showing the pipeline works — in production you would need thousands of documents for quality abstractive summaries.

---

## 8. How Evaluation Works

After generating summaries with all 5 models, the paper evaluates them using multiple metrics:

### Standard Metrics (computed automatically)

| Metric | What it measures |
|---|---|
| ROUGE-1 | Overlap of individual words (unigrams) |
| ROUGE-2 | Overlap of word pairs (bigrams) |
| ROUGE-L | Longest common subsequence of words |
| BERTScore | Semantic similarity using BERT embeddings (F1) |

### Intent-Based Metric (the paper's contribution)

1. Run JointBERT on the generated summary → extract predicted intent phrases
2. Run JointBERT on the gold summary → extract gold intent phrases
3. Compute precision/recall/F1 of predicted phrases vs. gold phrases
4. Also compute: does the summary correctly classify the case category?

**Why this matters:**  
The paper shows that ROUGE scores and human preference scores do not always agree — a summary can have high ROUGE but be rated poorly by humans because it misses the key intent. The intent-based metric agrees with human judgments more often.

### Human Evaluation

The paper also ran a human evaluation study:
- Annotators (law students) were shown summaries from different models
- Asked to rate on: informativeness, coherence, intent coverage
- The intent-based metric was compared to ROUGE in terms of correlation with human ratings
- Result: intent-based metric had higher Kendall's tau (rank correlation) with human ratings

---

## 9. JointBERT — The Intent Extractor

### What is JointBERT?

JointBERT solves two tasks simultaneously:
1. **Intent Classification:** Given a sentence/document, what category is it? (Murder, Robbery, etc.)
2. **Slot Filling (NER):** Given a sentence, which tokens are intent phrases?

It does both in one forward pass through BERT — that is why it is called "Joint".

### Input Format (the `JointBert_input_dataset_indian/` folder)

JointBERT expects three files per split (train/dev/test):

**`seq.in`** — one sentence per line (the raw text input):
```
revenue tribunal has held the petitioner to be eligible to be a tenant in respect of survey no 111 2 is the same for claiming her tenancy for survey no 114 2 also ...
```

**`seq.out`** — BIO tags for each token in the sentence above:
```
O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-LandDispute I-LandDispute I-LandDispute I-LandDispute I-LandDispute I-LandDispute I-LandDispute O O ...
```

BIO tagging explained:
- `O` = this token is NOT part of an intent phrase ("Outside")
- `B-Murder` = this token BEGINS a Murder intent phrase
- `I-Murder` = this token is INSIDE (continuation of) a Murder phrase
- Same pattern for LandDispute, Robbery, Corruption

**`label`** — one intent class per line (document-level):
```
LandDispute
Murder
Robbery
Corruption
```

**`intent_label.txt`** — list of all valid intent classes:
```
UNK
LandDispute
Murder
Corruption
Robbery
```

**`slot_label.txt`** — list of all valid slot tags:
```
PAD
UNK
O
B-Robbery
B-LandDispute
B-Murder
B-Corruption
I-Robbery
I-LandDispute
I-Murder
I-Corruption
```

### How to train JointBERT

The JointBERT model itself is not in this repo — only the formatted dataset is. To train it, use the [JointBERT repository](https://github.com/monologg/JointBERT):

```bash
git clone https://github.com/monologg/JointBERT
cd JointBERT
# Copy the JointBert_input_dataset_indian folder into data/
python main.py --task intent_slot \
               --model_type bert \
               --model_dir_or_name bert-base-uncased \
               --data_dir data/JointBert_input_dataset_indian \
               --do_train --do_eval
```

The trained model is then used in the demo app to extract intent phrases from new documents.

**Train/Dev/Test split:**
- Train: 1,385 sentences
- Dev: 146 sentences
- Test: 146 sentences

---

## 10. The Demo App

**Folder:** `demo_app_code/`  
**Technology:** Streamlit (Python web framework — write Python, get a web app)

### What the demo does

1. User pastes a legal document (or loads a sample) into the text box
2. User selects which summarization model to use (LetSum, Graphical, BERT, Legal-LED)
3. App generates the summary
4. App runs evaluation: ROUGE, BERTScore, and intent-based metric
5. Displays results with charts

### How to run locally

```bash
cd demo_app_code
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

### Key files in the app

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app — UI and evaluation pipeline |
| `graphicalModel.py` | The graphical summarizer (copy from `summarization_models/`) |
| `letsum.py` | The LetSum summarizer |
| `crf_alltrain.model` | Pre-trained CRF weights |
| `config.json` | JointBERT model configuration |
| `data_loader.py`, `trainer.py`, `utils.py` | JointBERT inference helpers |
| `intent_label.txt`, `slot_label.txt` | Label definitions for JointBERT |

---

## 11. Running Everything on Google Colab

Each notebook is self-contained — it clones this repo and installs its own dependencies. You do not need to upload any files manually.

### Steps (same for all notebooks)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub tab**
3. Paste: `Aditya-dom/legaldocsvalution`
4. Click the notebook you want
5. For `legal-led.ipynb` only: **Runtime → Change runtime type → T4 GPU → Save**
6. **Runtime → Run all**

### What each notebook does on Colab

| Notebook | Runtime | Time estimate | What it produces |
|---|---|---|---|
| `Bert/Extractive_summary_bert.ipynb` | CPU | ~30 min | 93 `.txt` summary files → `bert_summaries.zip` |
| `Graphical_Model.ipynb` | CPU | ~40 min | 93 × 3 fractions = 279 files |
| `Case_Summarizer.ipynb` | CPU | ~20 min | 93 `.txt` files at 30% length |
| `legal-led.ipynb` | **T4 GPU** | ~90 min | 93 `.txt` abstractive summaries |
| `Abstractive summarization/abstractive_summarization.ipynb` | CPU/GPU | varies | Seq2seq model training + predictions |

### Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'TransfoXLModel'` | transformers ≥ 5.0 removed this | Already fixed — notebook pins `transformers<5.0` |
| `FileNotFoundError: ../../summary_outputs/...` | Output dir not created | Already fixed — notebook pre-creates dirs |
| `CUDA out of memory` | GPU memory exhausted | In `legal-led.ipynb`, reduce `num_beams=1` |
| `ModuleNotFoundError: graphicalModel` | Wrong working directory | Make sure the `%cd` cell ran before the import cell |
| `python2: command not found` | Case_Summarizer used Python 2 | Already fixed — notebook uses `python3` |

---

## 12. Glossary

| Term | What it means |
|---|---|
| **Extractive summarization** | Picking existing sentences from the document. Output is always real text from the source. |
| **Abstractive summarization** | Generating new sentences that may not appear in the source. Can rephrase and condense. |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation. Counts word/phrase overlap between generated and reference summaries. Score between 0 and 1. |
| **BERTScore** | Uses BERT to compare meaning instead of exact words. Better at catching paraphrases than ROUGE. |
| **BERT** | Bidirectional Encoder Representations from Transformers. A large pre-trained language model by Google (2018). |
| **Transformer** | The neural network architecture behind BERT, GPT, and most modern NLP models. Uses attention mechanisms instead of recurrence. |
| **LSTM** | Long Short-Term Memory. A type of recurrent neural network that processes sequences step by step. Older than transformers but still used. |
| **CRF** | Conditional Random Field. A probabilistic model for sequence labeling (e.g., tagging which tokens are important). |
| **BIO tagging** | Begin-Inside-Outside. A labeling scheme for named entity recognition. B = start of entity, I = continuation, O = not an entity. |
| **Intent** | In this paper: the key phrases in a legal document that indicate the case type and what happened. |
| **Slot filling** | In NLP: identifying and labeling specific pieces of information in text (like filling in a form). Here: finding intent phrases. |
| **Intent classification** | Determining the overall category/purpose of a document or sentence. Here: Murder, Robbery, Land Dispute, or Corruption. |
| **JointBERT** | A model that does both intent classification and slot filling in a single BERT forward pass. |
| **Longformer / LED** | A transformer variant that handles long documents (up to 16,384 tokens) using sparse attention. |
| **Beam search** | A decoding strategy for generative models. Keeps the top-K candidate sequences at each step instead of committing to one word at a time. |
| **TF-IDF** | Term Frequency–Inverse Document Frequency. A score for how important a word is to a specific document relative to a collection. |
| **K-means clustering** | An algorithm that groups similar items together. Used in BERT summarizer to cluster sentences by meaning. |
| **Streamlit** | A Python library that turns Python scripts into interactive web apps with minimal code. |
| **LREC** | Language Resources and Evaluation Conference. A major NLP conference held every 2 years. |
| **Rhetorical role** | The function a sentence plays in a legal document: stating facts, making arguments, citing precedent, announcing the holding, etc. |
| **Seq2seq** | Sequence-to-sequence. A model architecture that maps an input sequence to an output sequence, used for translation, summarization, etc. |
| **Attention mechanism** | A way for neural networks to "focus" on specific parts of the input when producing each output token. Core component of all transformers. |
| **Kendall's tau** | A statistical measure of rank correlation. Used in the paper to compare how well different metrics agree with human rankings. |
| **Transfer learning** | Using a model trained on one task/domain as a starting point for another. Here: training on Indian data, running on Australian data. |
