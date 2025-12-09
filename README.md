# wp3-hackathon

Hackathon for Work Package 3, with representatives from Västra Götalandsregionen, Region Stockholm and Region Skåne.

## Background

For this Hackathon we will try cocreating a set of agents to help with onboarding a new employee. Your creativity is the
limit as to what these agents should be able to solve but make sure to let others know what you're working on so we can
make sure to create agents with different capabilities.

The real challenge and end goal of the project is to let these agents interact by using [Google's A2A protocol](https://a2a-protocol.org/latest/).
This means that you can (and are enouraged to) create agents in *any framework* (Javascript, Python, C# etc.). What
you will need to do however is make them compatibel with A2A.

At least one group will att some point have create an AgentExecutor (and some way to see what it does) to handle
interfacing with the agents we've created.

## Python development setup

To help you get started (in Python) Isak has created a few helpful examples on how agents can be built in the LangGraph
framework, see [/notebooks](/notebooks/).

### Installation

Clone repo:

```bash
git clone https://github.com/isak-b/wp3-hackathon.git
cd wp3-hackathon
```

Install python dependencies:

- Option 1: Manually create Virtual Environment:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

  And install requirements:
  
  ```bash
  pip install -r requirements.txt
  ```

- Option 2: Use UV

  ```bash
  uv sync
  ```

### Configure the Azure OpenAI API

Export the following environment variables:

```bash
export OPENAI_BASE_URL="https://<resource>.cognitiveservices.azure.com/openai/v1"
export OPENAI_API_KEY="your azure openai api key (see notes)"
```

or create a `.env` file and add them there (see [.env.example](.env.example)).

NOTE:

- Note that your Azure API-key and endpoint depends on what Azure subscription you use.
- Ask Isak or whoever manages your Azure subscription for a key and endpoint.

#### Recommended Azure OpenAI models

| Use case                   | Model(s)               |
| -------------------------- | ---------------------- |
| Text                       | gpt-4o, gpt-5.1        |
| Reasoning                  | o1, o1-mini, o3-mini   |
| Text-to-Image              | dall-e-3               |
| Image-to-Text              | gpt-4o, gpt-4o-mini    |
| Text-to-Speech             | tts, tts-hd            |
| Speech-to-Text             | whisper                |
| Text Embeddings/Vectorizer | text-embedding-ada-002 |

NOTE:

- Note that what models you have available depend on what Azure subscription you use.
- Ask Isak or whoever manages your Azure subscription for specific model recommendations and their names.

## Examples

See `notebooks/` for examples on how to do different things with langchain and langgraph.

Note that the examples were not created for this hackathon specifically, so they may or may not be relevant.

Also see [Isak's LLM notebooks](https://github.com/isak-b/llm-notebooks) for more examples.
