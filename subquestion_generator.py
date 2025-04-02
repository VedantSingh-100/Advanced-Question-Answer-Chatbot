import json
from typing import List
from enum import Enum

from instructor import OpenAISchema
from pydantic import Field, create_model
from openai_utils import llm_call


# DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
#                  You are an AI agent that takes a complex user question and returns a list of simple subquestions to answer the user's question.
#                  You are provided a set of functions and data sources that you can use to answer each subquestion.
#                  If the user question is simple, just return the user question, the function, and the data source to use.
#                  You can only use the provided functions and data sources.
#                  The subquestions should be complete questions that can be answered by a single function and a single data source.
#                  """

# DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
#     You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
#     When presented with a complex user question, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original query.
#     You have at your disposal a pre-defined set of functions and data sources to utilize in answering each sub-question.
#     If a user question is straightforward, your task is to return the original question, identifying the appropriate function and data source to use for its solution.
#     Please remember that you are limited to the provided functions and data sources, and that each sub-question should be a full question that can be answered using a single function and a single data source.
# """

DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
    You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
    You have at your disposal a pre-defined set of functions and files to utilize in answering each sub-question.
    Please remember that your output should only contain the provided function names and file names, and that each sub-question should be a full question that can be answered using a single function and a single file.
"""

DEFAULT_USER_TASK = ""


class FunctionEnum(str, Enum):
    """The function to use to answer the questions.
    Use vector_retrieval for fact-based questions such as demographics, sports, arts and culture, etc.
    Use llm_retrieval for summarization questions, such as positive aspects, history, etc.
    """

    VECTOR_RETRIEVAL = "vector_retrieval"
    LLM_RETRIEVAL = "llm_retrieval"


# def generate_subquestions(
#     question,
#     file_names: List[str],
#     system_prompt="You are a subquestion generator.",
#     user_task="You are a chatbot for Palantir Technologies, answering various queries about the company, products, and job postings.",
#     llm_model="gpt-3.5-turbo",
# ):
#     """
#     Generates a list of subquestions from a user question along with the
#     file name and the function to use to answer the question using OpenAI LLM.
#     - 'file_names': a list of possible document names (e.g. ["palantir_overview","palantir_jobs","palantir_philanthropy"])
#     - 'FunctionEnum': your enumerated type for "vector_retrieval" or "llm_retrieval".
#     """

#     # 1) Dynamically create an Enum from file_names
#     valid_file_names = file_names
#     FilenameEnum = Enum("FilenameEnum", {x.upper(): x for x in valid_file_names})

#     # 2) Create pydantic classes
#     QuestionBundle = create_model(
#         "QuestionBundle",
#         question=(str, Field(
#             None,
#             description="The subquestion extracted from the user's query."
#         )),
#         function=(FunctionEnum, Field(
#             None,
#             description="The function to use: vector_retrieval or llm_retrieval."
#         )),
#         file_name=(FilenameEnum, Field(
#             None,
#             description="Which document to use to answer this subquestion."
#         )),
#     )

#     SubQuestionBundleList = create_model(
#         "SubQuestionBundleList",
#         subquestion_bundle_list=(
#             List[QuestionBundle],
#             Field(
#                 None,
#                 description="A list of subquestions for the user's query."
#             ),
#         ),
#         __base__=OpenAISchema,
#     )

#     # 3) Build user prompt
#     user_prompt = f"{user_task}\nUser's question: {question}"

#     # 4) Provide new few_shot_examples
#     few_shot_examples = [
#         {
#             "role": "user",
#             "content": "Give me a brief overview of Palantir's main products.",
#         },
#         {
#             "role": "function",
#             "name": "SubQuestionBundleList",
#             "content": """
#             {
#               "subquestion_bundle_list": [
#                 {
#                   "question": "What are Palantir's main products?",
#                   "function": "vector_retrieval",
#                   "file_name": "PALANTIR_OVERVIEW"
#                 }
#               ]
#             }
#             """,
#         },
#         {
#             "role": "user",
#             "content": "Which roles are available for entry-level engineers in the New York office?",
#         },
#         {
#             "role": "function",
#             "name": "SubQuestionBundleList",
#             "content": """
#             {
#               "subquestion_bundle_list": [
#                 {
#                   "question": "Which open roles mention entry-level or junior engineering positions in New York?",
#                   "function": "vector_retrieval",
#                   "file_name": "PALANTIR_JOBS"
#                 }
#               ]
#             }
#             """,
#         },
#         {
#             "role": "user",
#             "content": "Summarize Palantir's philanthropic initiatives and any major charitable contributions they've made recently.",
#         },
#         {
#             "role": "function",
#             "name": "SubQuestionBundleList",
#             "content": """
#             {
#               "subquestion_bundle_list": [
#                 {
#                   "question": "What philanthropic efforts does Palantir have?",
#                   "function": "llm_retrieval",
#                   "file_name": "PALANTIR_PHILANTHROPY"
#                 }
#               ]
#             }
#             """,
#         },
#         {
#             "role": "user",
#             "content": "What is Palantir's mission statement, and do they have any internship roles available in London?",
#         },
#         {
#             "role": "function",
#             "name": "SubQuestionBundleList",
#             "content": """
#             {
#               "subquestion_bundle_list": [
#                 {
#                   "question": "What is Palantir's mission statement?",
#                   "function": "vector_retrieval",
#                   "file_name": "PALANTIR_OVERVIEW"
#                 },
#                 {
#                   "question": "List Data Science internship roles in London.",
#                   "function": "vector_retrieval",
#                   "file_name": "PALANTIR_JOBS"
#                 }
#               ]
#             }
#             """,
#         },
#     ]

#     # 5) Call your LLM with the function schema
#     response, cost = llm_call(
#         model=llm_model,
#         function_schema=[SubQuestionBundleList.openai_schema],
#         output_schema={"name": SubQuestionBundleList.openai_schema["name"]},
#         system_prompt=system_prompt,
#         user_prompt=user_prompt,
#         few_shot_examples=few_shot_examples,
#     )

#     # 6) Parse out the function call result
#     # The LLM should respond with a "function_call" that has arguments in JSON
#     subquestions_list = json.loads(response.choices[0].message.function_call.arguments)
#     # Then parse that JSON into the pydantic model
#     subquestions_pydantic_obj = SubQuestionBundleList(**subquestions_list)
#     # Extract the actual list of subquestions
#     subquestions_list = subquestions_pydantic_obj.subquestion_bundle_list

#     return subquestions_list, cost

def generate_subquestions(
    question: str,
    file_names: List[str],
    system_prompt: str = "You are a subquestion generator.",
    user_task: str = (
        "You are an AI assistant that helps candidates find and understand "
        "job postings at Palantir. You only have access to CSV files that each "
        "contain chunked text + metadata for a single Palantir job posting. "
        "Your job is to figure out which file(s) and which retrieval function to use "
        "to answer the user's question."
    ),
    llm_model: str = "gpt-3.5-turbo",
):
    """
    Generates a list of subquestions from a user's question along with
    the file name(s) and the function to use to answer the question, using OpenAI LLM.

    - 'file_names': a list of possible document names
      (e.g. ["PALANTIR_JOBS_1", "PALANTIR_JOBS_2", ..., "PALANTIR_JOBS_84"]).
    - 'FunctionEnum': an enum with "vector_retrieval" or "llm_retrieval".

    Returns:
      (subquestions_list, cost)
      - subquestions_list: list of pydantic-validated subquestions with structure:
        [
          {
            "question": "...",
            "function": "vector_retrieval" or "llm_retrieval",
            "file_names": ["PALANTIR_JOBS_12", "PALANTIR_JOBS_13", ...]
          },
          ...
        ]
      - cost: an approximate token cost (depending on your LLM usage tracking)
    """

    # ---------------------------------------------------------------------
    # 1) Dynamically create an Enum from the file_names
    # ---------------------------------------------------------------------
    valid_file_names = file_names  # e.g. ["PALANTIR_JOBS_1", ..., "PALANTIR_JOBS_84"]
    FilenameEnum = Enum("FilenameEnum", {x.upper(): x for x in valid_file_names})

    # ---------------------------------------------------------------------
    # 2) Create pydantic classes for subquestions & the top-level container
    # ---------------------------------------------------------------------
    QuestionBundle = create_model(
        "QuestionBundle",
        question=(str, Field(
            ...,
            description="The subquestion extracted from the user's query."
        )),
        function=(FunctionEnum, Field(
            ...,
            description="The function to use: vector_retrieval or llm_retrieval."
        )),
        file_names=(List[FilenameEnum], Field(
            ...,
            description="Which document(s) to use to answer this subquestion."
        )),
    )

    SubQuestionBundleList = create_model(
        "SubQuestionBundleList",
        subquestion_bundle_list=(
            List[QuestionBundle],
            Field(
                ...,
                description="A list of subquestions for the user's query."
            ),
        ),
        __base__=OpenAISchema,
    )

    # ---------------------------------------------------------------------
    # 3) Build the user prompt
    # ---------------------------------------------------------------------
    user_prompt = f"{user_task}\nUser's question: {question}"

    # ---------------------------------------------------------------------
    # 4) Provide a few-shot conversation history (job-focused only)
    # ---------------------------------------------------------------------
    # This helps the LLM see how we want the final JSON structured
    # and how to decide on function usage + doc names.
    few_shot_examples = [
        # Example 1: Filter by location and seniority
        {
            "role": "user",
            "content": "I'm looking for senior engineering roles in London. What do you have?"
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "Which open roles mention senior or lead engineering positions in London?",
                  "function": "vector_retrieval",
                  "file_names": ["PALANTIR_JOBS_1", "PALANTIR_JOBS_2", "PALANTIR_JOBS_3", "PALANTIR_JOBS_4", "..."]
                }
              ]
            }
            """
        },

        # Example 2: Ask about responsibilities and job description
        {
            "role": "user",
            "content": "What are the responsibilities for a Data Scientist role at Palantir?"
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "What do the job descriptions say about responsibilities for Data Scientist positions?",
                  "function": "vector_retrieval",
                  "file_names": [
                    "PALANTIR_JOBS_1", 
                    "PALANTIR_JOBS_2", 
                    "PALANTIR_JOBS_3", 
                    "PALANTIR_JOBS_4", 
                    "PALANTIR_JOBS_5",
                    "...",
                  ]
                }
              ]
            }
            """
        },

        # Example 3: Summarizing multiple postings
        {
            "role": "user",
            "content": "Can you summarize the different internship roles offered in the US?"
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "Summarize internship positions for US-based roles.",
                  "function": "llm_retrieval",
                  "file_names": [
                    "PALANTIR_JOBS_1",
                    "PALANTIR_JOBS_2",
                    "PALANTIR_JOBS_3",
                    "PALANTIR_JOBS_4",
                    "PALANTIR_JOBS_5",
                    "...",
                  ]
                }
              ]
            }
            """
        }
    ]

    # ---------------------------------------------------------------------
    # 5) Call your LLM with the function schema
    # ---------------------------------------------------------------------
    response, cost = llm_call(
        model=llm_model,
        function_schema=[SubQuestionBundleList.openai_schema],
        output_schema={"name": SubQuestionBundleList.openai_schema["name"]},
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shot_examples=few_shot_examples,
    )

    # ---------------------------------------------------------------------
    # 6) Parse the JSON response into Pydantic
    # ---------------------------------------------------------------------
    # The LLM should respond with JSON that matches SubQuestionBundleList
    # e.g. {"subquestion_bundle_list": [ {...}, {...} ]}
    subquestions_json = json.loads(response.choices[0].message.function_call.arguments)

    # Validate to ensure correct structure & valid file names
    subquestions_pydantic_obj = SubQuestionBundleList(**subquestions_json)

    # Final list of subquestions
    subquestions_list = subquestions_pydantic_obj.subquestion_bundle_list

    return subquestions_list, cost


