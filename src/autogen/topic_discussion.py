## Conversation between 3 agents with sequential workflow
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

topic_agent = ConversableAgent(
    name="topic_agent",
    system_message="""You are responsible for suggesting a discussion topic.
                   Please present one socially important issue currently relevant """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

economic_agent = ConversableAgent(
    name="economic_agent",
    system_message= """You are an economist.
                    Please provide your opinion on the given topic from an economic perspective. """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

social_agent = ConversableAgent(
    name="social_agent",
    system_message= """You are an social expert.
                    Please provide your opinion on the given topic from an social perspective, considering previous opinions. """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

environmental_agent = ConversableAgent(
    name="social_agent",
    system_message= """You are an environmental expert.
                    Please provide your opinion on the given topic from an environmental perspective, considering previous opinions. """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

ethical_agent = ConversableAgent(
    name="ethical_agent",
    system_message= """You are an ethicist.
                    Please provide your opinion on the given topic from an ethical perspective, considering the previous opinions. """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)



topic = "Sports culture"
chat_result = topic_agent.initiate_chats(
    [
        {
            "recipient": economic_agent,
            "message": f"Please provide your opinion on the following topic from an economic perspective.: {topic}",
            "max_turns": 2,
            "summary_method" : "last_msg", # summrizing the most recent msg
        },
        {
            "recipient": social_agent,
            "message": f"Please provide your opinion on the following topic from an social perspective.: {topic}",
            "max_turns": 2,
            "summary_method" : "last_msg",
        },
        {
            "recipient": environmental_agent,
            "message": f"Please provide your opinion on the following topic from an environmental perspective.: {topic}",
            "max_turns": 2,
            "summary_method" : "last_msg",
        },
        {
            "recipient": ethical_agent,
            "message": f"Please provide your opinion on the following topic from an ethical perspective.: {topic}",
            "max_turns": 2,
            "summary_method" : "last_msg",
        },
    ]
)

print("First Chat Summary: ", chat_result[0].summary)
print("Second Chat Summary: ", chat_result[1].summary)
print("Third Chat Summary: ", chat_result[2].summary)
print("Fourth Chat Summary: ", chat_result[3].summary)

