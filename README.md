# py_gen_ai_lab


### Project Structure
```tree
.
├───assets
│   ├───docs
│   └───prompt_template
├───src
│   ├───Langchain
│   │   ├───00_UseCases
│   │   ├───01_Models
│   │   │   ├───1_LLMs
│   │   │   ├───2_ChatModels
│   │   │   └───3_EmbeddedModels
│   │   ├───02_Prompts
│   │   ├───03_StructuredOutput
│   │   ├───04_OutputParsers
│   │   ├───05_Chains
│   │   ├───06_Runnables
│   │   ├───07_DocumentLoaders
│   │   ├───08_TextSplitters
│   │   ├───09_VectorStores
│   │   ├───10_Retrievers
│   │   ├───11_RAG_RetrivevalAugmentedGeneration
│   │   ├───12_Tools
│   │   ├───13_ToolsCalling
│   │   └───14_Agent_E2E
└───tests
```

### Setup

1. Steps to create the virtual environment
    ```bash
    # Create virtual environment
    python -m venv .venv_genai
    # activate on windows
    .\.venv_genai\Scripts\activate
    # install packages as per requirement file
    pip install -r requirements.txt
    ```
2. Open VS code and install Jyputer extension and python extension if not installed already.
3. Create account on the following in case not available
    1. ChatGpt: Properitory tool
    2. HuggingFace: Repository for many open models.
4.  

### Execution on files

1. Open command prompt from base directory
2. activate environment
3. run file by passing full path