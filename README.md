# DeepResearch: a multi-agent LLM system for deep web research

Code and thesis text for my BSc thesis at the University of Zagreb, Faculty of Electrical Engineering and
Computing (FER): **"Multi-agent AI system for deep research"**
(*Višeagentski sustav umjetne inteligencije za dubinsko istraživanje*), June 2025.
Mentor: izv. prof. dr. sc. Mario Brčić.

DeepResearch answers open research questions by having several specialised LLM agents search the web,
check each other's work and write a sourced report. It is built with Microsoft
[AutoGen](https://github.com/microsoft/autogen) (AgentChat) on Google's Gemini 2.5 Pro.

## Architecture

Four agents share one group chat. A `SelectorGroupChat` uses the LLM to pick which agent speaks next,
based on the conversation so far, giving a **search → verify → summarize** loop:

| Agent | Role |
|---|---|
| `research_assistant` | Breaks the question into search queries, calls the Google search tool, analyses what it finds |
| `verifier` | Checks progress and gaps, suggests drill-downs and new angles, and replies `APPROVED` once the question is answered |
| `summary_agent` | Writes the final markdown report with all visited sources, then ends the run with `TERMINATE` |
| `user_proxy_agent` | Lets the user approve, redirect or add input during the run |

The run stops when the summary agent says `TERMINATE`, or after 30 messages.

**Custom search tool** (`google_search.py`): queries the Google Custom Search JSON API, then downloads each
result page, strips scripts and styles, resolves relative links, and converts the page to markdown
(capped at 15,000 characters per page) so the agent reads page content, not just snippets.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env      # then add your Gemini, Google Custom Search API key and search-engine ID
python app.py             # type a research question at the "Prompt:" line
```

Requires Python 3.10+, a [Gemini API key](https://aistudio.google.com/apikey), and a Google
[Programmable Search Engine](https://programmablesearchengine.google.com/) with a Custom Search API key.
The Gemini model is called through Google's OpenAI-compatible endpoint. The code was written in June 2025
against the `gemini-2.5-pro-preview-03-25` model; if Google has retired that preview, change the `model`
string in `app.py`.

## Results (from the thesis)

DeepResearch (DR) was compared with a GPT-4o-based reference implementation on six questions: two
differently worded question sets, each at three difficulty levels (F1 easy, F2 medium, F3 hard). The table
shows averages over the two sets.

| Level | Latency DR / GPT-4o | Report length (words) | Output tokens | Cost per query (USD) |
|---|---|---|---|---|
| F1 easy | **52 s** / 481 s | 285 / 654 | 375 / 825 | 0.011 / 0.025 |
| F2 medium | **163 s** / 539 s | 832 / 6,172 | 850 / 3,300 | 0.026 / 0.099 |
| F3 hard | **511 s** / 974 s | 1,490 / 13,119 | 1,200 / 4,550 | 0.036 / 0.136 |

- **Latency:** about 9× faster on easy questions, 3× on medium and 2× on hard ones. Extra verifier rounds
  eat part of the advantage as questions get harder.
- **Cost:** 55–75% fewer output tokens, about 3.6× cheaper per query on average.
- **Reports:** 2–9× shorter, with consistent headings and lists. On the hardest question the GPT-4o
  report gave market-share figures without references, while DR linked every figure to a source.
- **Robustness:** rewording the questions did not change the ranking.

**Limitations.** Each question was run once, so API and network jitter are not averaged out. Report
quality was judged by a single evaluator, and images and tables in sources were not measured.

## Repository layout

| Path | Contents |
|---|---|
| `app.py` | Agents, prompts, selector group chat and the console entry point |
| `google_search.py` | Google Custom Search tool with page fetching and HTML-to-markdown conversion |
| `Thesis_text/` | LaTeX source of the thesis (Croatian), bibliography and result figures (`architecture.pdf`, `latencija.pdf`, `duljina.pdf`, `propusnost.pdf`, `tokeni.pdf`) |

The official thesis assignment page is not included; the LaTeX source compiles without it.
