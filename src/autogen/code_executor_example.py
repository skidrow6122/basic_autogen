import os
import warnings
warnings.filterwarnings("ignore")

import autogen
from autogen.coding import LocalCommandLineCodeExecutor

from dotenv import load_dotenv
load_dotenv()

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    }
]

assistant_agent = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0  # Ensures high consistency in code generation with settings that make responses consistent
    }
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    max_consecutive_auto_reply=10, # When human_input_mode is set to 'NEVER', UserProxyAgent automatically responds, and this setting prevents infinite responses or code execution.
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="data")
    },
    human_input_mode="ALWAYS" # Agent always requires a human input
)

chat_result = user_proxy.initiate_chat(
    assistant_agent,
    message="""Please check with code what prime numbers exist up to 100.""",
    summary_method="reflection_with_llm",
)

print("Chat history:", chat_result.chat_history)
print("-"*50)
print("Summary:", chat_result.summary)
print("-"*50)
print("Cost info:", chat_result.cost)


# To send additional messages continuing from an existing conversation, send another message to the assistant appending to the existing conversation thread.
user_proxy.send(
    recipient=assistant_agent,
    message="""메이저리그에 대한 간략한 설명문을 만들고, 키워드를 추출하세요"""
)

