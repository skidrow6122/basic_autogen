## Conversation in group chat
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager


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


group_chat = GroupChat(
    agents=[topic_agent, economic_agent, social_agent, environmental_agent, ethical_agent],
    messages=[],
    max_round=6,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

chat_result = topic_agent.initiate_chat(
    group_chat_manager,
    message="Please talk about gambling",
    summary_method="reflection_with_llm"
)

print("Group Chat Summary: ", chat_result.summary)
