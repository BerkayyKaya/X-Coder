<div align="center">

# X-Coder<br>Multi-Agent LLM Coding Assistant

![Python 3.10.0](https://img.shields.io/badge/Python-3.10.0-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Transformers](https://img.shields.io/badge/Transformers-🤗_Hugging_Face-FFD21E?style=for-the-badge&logoColor=black)
![llama-cpp-python](https://img.shields.io/badge/llama.cpp--python-black?style=for-the-badge&logo=python)

</div>

<br>

X-Coder is a localized, multi-agent AI coding assistant powered by Small Language Models. Instead of relying on a single massive model, X-Coder utilizes a custom-built orchestration engine to manage an ensemble of specialized models that communicate with each other to plan, write, and verify code efficiently.

## 🧠 System Architecture

The project revolves around a dynamic workflow managed by a central router. The system consists of 5 specialized agent roles:

*   **Router Agent:** The brain of the operation. It analyzes the user's input and intelligently routes the request to the appropriate agent (`PLANNER`, `CODER`, `TESTER`, or `CHATTER`).
*   **Planner Agent:** Breaks down complex user requests into smaller, manageable tasks. It generates a detailed step-by-step execution plan and forwards it to the Coder.
*   **Coder Agent:** The core developer. If a plan exists, it strictly follows it to generate the code. If no plan was provided, it autonomously creates a lightweight plan on the fly, writes the code, and hands it off to the Tester.
*   **Tester Agent (WIP):** Responsible for code quality. It verifies the syntax and functionality of the generated code. If the code fails, it triggers a feedback loop, sending the workflow back to the Coder for revisions.
*   **Chatter Agent:** The user-facing communicator. If code was generated via a plan, it provides a concise summary to the user. If the user is just asking general questions, it acts as a mentor and provides conversational answers.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) Currently used for rapid prototyping; planned to be replaced in future iterations.
*   **Backend & Orchestration:** [FastAPI](https://fastapi.tiangolo.com/) with completely custom-built workflow orchestration 
*   **LLM Infrastructure:** Highly flexible local inference supporting both `transformers` and `llama.cpp-python`, allowing the system to run efficiently across different hardware setups.

## 🚧 Current Limitations & Roadmap

This project is in active development. The following features are currently missing and are planned for future updates:

- [ ] **Tester Agent Implementation:** The automated code verification loop is currently under development.
- [ ] **Chat History:** The system currently does not retain previous conversation history within a single session.
- [ ] **Long-Term Memory:** Missing long-term state retention to remember user preferences and past project contexts.
- [ ] **Tool / Function Calling:** The agents do not yet have the capability to execute external tools.

## 🚀 Getting Started (Soon)

*(Note: Specific SLMs are currently undergoing active testing and benchmarking. Model configuration instructions will be updated soon.)*