\# Kuku Promo Finder



Finds and ranks 20–60 second promo cuts from timestamped microdrama screenplays. The pipeline parses screenplay events, generates candidate windows, shortlists them through semantic retrieval, and uses a local Ollama model to score hook strength, standalone clarity, and cliffhanger quality.



\## Setup



From the project folder in Windows CMD:



```bat

python -m venv .venv

.venv\\Scripts\\activate

python -m pip install -r requirements.txt

ollama pull qwen2.5:3b

```



Create a `.env` file in the project root:



```dotenv

LLM\_PROVIDER=ollama

LLM\_MODEL=qwen2.5:3b

```



Start Ollama and place the supplied screenplay archive at `data\\screenplays.zip`.



\## Run



```bat

python main.py data\\screenplays.zip

python windows.py

python retrieval.py

python judge.py --all

```



The ranked promos are written to `outputs\\ranked.csv`. Judge decisions and validation attempts are logged in `outputs\\judge\_attempts.jsonl`.



\## Results



\- Full run: 50 episodes processed, 139 ranked promos across 47 episodes, and zero validation failures.

\- Three-episode evaluation: Shortlisting reduced judged windows from 120 to 4. Final matches against three manually reviewed reference picks fell from 2 to 0.

\- Episode 1 refinement: The original cut was retained after three alternative judgments. The best alternative tied the original at 7/9.



The rankings use screenplay events. An editor should review the actual footage before publishing a cut.

