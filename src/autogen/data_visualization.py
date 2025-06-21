import os
import warnings

from autogen.coding import LocalCommandLineCodeExecutor
from autogen import GroupChat
from autogen import GroupChatManager
#from src.autogen.group_chat import group_chat

warnings.filterwarnings('ignore')

import autogen
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image

from dotenv import load_dotenv
load_dotenv()

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 42 #deprecated setting
}

##################################
######## agent definition ########
##################################
# user-proxy agent - code execution
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A human administrator",
    code_execution_config= {
        "executor": LocalCommandLineCodeExecutor(work_dir="data")
    },
    human_input_mode="NEVER"
    ## max_consecutive_auto_reply=10,
)

## coder agent - code writing
coder = autogen.AssistantAgent(
    name="coder",
    llm_config=llm_config,
)

## critic agent - code assessment
critic = autogen.AssistantAgent(
    name="critic",
    system_message="""A critic who is very skilled in computer programming.
    You are a highly skilled assistant tasked with evaluating the quality of a given data visualization code, providing a score from 1 (poor) to 10 (excellent) with clear justification. 
    Each evaluation must consider best practices for data visualization and assess the code carefully across the following dimensions:
     - Bug (Bug): Are there any bugs, syntax errors, or typos? Does the code fail to compile? If so, why? How should it be corrected? If a bug exists, the bug score must be below 5.
     - Data Transformation (Transformation): Has the data been properly transformed for the visualization type? For example, was the dataset appropriately filtered, aggregated, or grouped where necessary? If a date field is used, has it been converted to a date object first?
     - Goal Compliance (Compliance): How well does the code achieve the specified visualization objective?
     - Visualization Type (Type): Considering best practices, is the chosen visualization type appropriate for the data and the intended message? If a more effective chart type exists, the score must be below 5.
    - Data Encoding (Encoding): Is the data encoded appropriately for the visualization type?
    - Aesthetics (Aesthetics): Are the visual aesthetics suitable for the visualization type and the data?

    A score should be provided for each criterion:
    (Bug: 0, Transformation: 0, Compliance: 0, Type: 0, Encoding: 0, Aesthetics: 0)
    Do not suggest any code.
    Finally, based on the above critique, provide a specific list of recommended actions the coder should take to improve the code.
    """,
    llm_config=llm_config,
)

##################################
####### group chat setting #######
##################################
group_chat = autogen.GroupChat(
    agents=[user_proxy, coder, critic],
    messages=[],
    max_round=6
)
## group chat manager agent
group_chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

##################################
############ Execution ###########
##################################
user_proxy.initiate_chat(
    group_chat_manager,
    message="""Please load the data from https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv?raw=true,
             then create a chart showing the relationship between the age and pclass variables.
             Save the chart as an image file.
             Before creating the chart, print the dataset’s column names for verification.""",
)

Image(filename="data/age_vs_pclass_boxplot.png")
