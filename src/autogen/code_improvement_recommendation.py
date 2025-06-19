## Conversation between 2 agents workflow
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

from autogen import ConversableAgent
#from langchain_openai import ChatOpenAI

llm_config = {
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "api_key": os.environ["OPENAI_API_KEY"],
            }
        ]
}

coder_agent = ConversableAgent(
    name="junior_coder_Agent",
    system_message="""You are a third-year Python code specialist and software engineer.
                     If there`s anything you don't understand, ask the Senior Coder Agent. """,
    llm_config=llm_config
)
manager_agent = ConversableAgent(
    name="senior_coder_agent",
    system_message= """You are a Python code specialist and software engineer with 20 years of experience.
                    When a question is given, answer it with expertise.
                    If code is provided, review it and explore ways to improve its efficiency. """,
    llm_config=llm_config
)

chat_result = coder_agent.initiate_chat(
    manager_agent,
    message="""Please explain the Fibonacci sequence code.""",
    summary_method="reflection_with_llm",
    max_turns=2, #maximum conversation pair
)

print(chat_result.summary)
print(ConversableAgent.DEFAULT_SUMMARY_PROMPT)

