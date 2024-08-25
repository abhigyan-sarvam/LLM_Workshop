# LLM OS

Lets build the LLM OS proposed by Andrej Karpathy [in this tweet](https://twitter.com/karpathy/status/1723140519554105733), [this tweet](https://twitter.com/karpathy/status/1707437820045062561) and [this video](https://youtu.be/zjkBMFhNj_g?t=2535).

## The LLM OS design:
- LLMs are the kernel process of an emerging operating system.
- This process (LLM) can solve problems by coordinating other resources (memory, computation tools).
- The LLM OS:
  - [x] Can read/generate text
  - [x] Has more knowledge than any single human about all subjects
  - [x] Can browse the internet
  - [x] Can use existing software infra (calculator, python, mouse/keyboard)
  - [ ] Can see and generate images and video
  - [ ] Can hear and speak, and generate music
  - [ ] Can think for a long time using a system 2
  - [ ] Can “self-improve” in domains
  - [ ] Can be customized and fine-tuned for specific tasks
  - [x] Can communicate with other LLMs

## Running the LLM OS:

### 1. Create a conda environment (or virtual environment)

```shell
conda create -n llm-workshop python=3.11
conda activate llm-workshop
```

### 2. Install libraries

```shell
pip install -r requirements.txt
```

### 3. Export credentials

- Export your OpenAI API Key

```shell
export OPENAI_API_KEY=***
```

- Export your EXA_API_KEY (get it from [here](https://dashboard.exa.ai/api-keys))

```shell
export EXA_API_KEY=***
```

### 4. Run PgVector

We use PgVector to provide long-term memory and knowledge to the LLM OS.
Please install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) and run PgVector using either the helper script or the `docker run` command.

- Run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```
### 5. Run the LLM OS App

```shell
streamlit run app.py
```

- Open [localhost:8501](http://localhost:8501) to view your LLM OS.
- Add a blog post to knowledge base: https://blog.samaltman.com/gpt-4o
- Ask: What is gpt-4o?
- Web search: Whats happening in Dharwad?
- Calculator: Add 5+10!
- Enable the Research Assistant and ask: write a report on audio-conditioned Large Language Model


#### Note: A large part of this repo is either borrowed or inspired from (phidata)[https://github.com/phidatahq/phidata].