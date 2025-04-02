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
        "You are a chatbot for Palantir Technologies, answering various queries "
        "about the company, products, and job postings."
    ),
    llm_model: str = "gpt-3.5-turbo",
):
    """
    Generates a list of subquestions from a user question along with the
    file name and the function to use to answer the question using OpenAI LLM.
    
    - 'file_names': a list of possible document names 
      (e.g. ["PALANTIR_OVERVIEW", "PALANTIR_PHILANTHROPY", "PALANTIR_JOBS_1", ...]).
    - 'FunctionEnum': your enum with "vector_retrieval" or "llm_retrieval".
    """

    # ---------------------------------------------------------------------
    # 1) Dynamically create an Enum from the file_names *only*.
    #    (No more ["PALANTIR_JOBS"] in the list.)
    # ---------------------------------------------------------------------
    valid_file_names = file_names  # e.g. ["PALANTIR_JOBS_1", ..., "PALANTIR_JOBS_84"]
    FilenameEnum = Enum("FilenameEnum", {x.upper(): x for x in valid_file_names})
    # FilenameEnum = (List[FilenameEnum], {x.upper(): x for x in valid_file_names})

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
            description="Which document to use to answer this subquestion."
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
    # 4) Provide a few-shot conversation history (updated to use PALANTIR_JOBS_1, etc.)
    # ---------------------------------------------------------------------
    few_shot_examples = [
        # Example 1: Overview
        {
            "role": "user",
            "content": "Give me a brief overview of Palantir's main products.",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "What are Palantir's main products?",
                  "function": "vector_retrieval",
                  "file_names": "PALANTIR_OVERVIEW"
                }
              ]
            }
            """,
        },

        # Example 2: Entry-level job postings
        {
            "role": "user",
            "content": "Which roles are available for entry-level engineers in the New York office?",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "Which open roles mention entry-level or junior engineering positions in New York?",
                  "function": "vector_retrieval",
                  "file_names": ["PALANTIR_JOBS_1", "PALANTIR_JOBS_2", "PALANTIR_JOBS_3", "PALANTIR_JOBS_4", "PALANTIR_JOBS_5", "PALANTIR_JOBS_6", "PALANTIR_JOBS_7", "PALANTIR_JOBS_8", "PALANTIR_JOBS_9", "PALANTIR_JOBS_10", "PALANTIR_JOBS_11", "PALANTIR_JOBS_12", "PALANTIR_JOBS_13", "PALANTIR_JOBS_14", "PALANTIR_JOBS_15", "PALANTIR_JOBS_16", "PALANTIR_JOBS_17", "PALANTIR_JOBS_18", "PALANTIR_JOBS_19", "PALANTIR_JOBS_20", "PALANTIR_JOBS_21", "PALANTIR_JOBS_22", "PALANTIR_JOBS_23", "PALANTIR_JOBS_24", "PALANTIR_JOBS_25", "PALANTIR_JOBS_26", "PALANTIR_JOBS_27", "PALANTIR_JOBS_28", "PALANTIR_JOBS_29", "PALANTIR_JOBS_30", "PALANTIR_JOBS_31", "PALANTIR_JOBS_32", "PALANTIR_JOBS_33", "PALANTIR_JOBS_34", "PALANTIR_JOBS_35", "PALANTIR_JOBS_36", "PALANTIR_JOBS_37", "PALANTIR_JOBS_38", "PALANTIR_JOBS_39", "PALANTIR_JOBS_40", "PALANTIR_JOBS_41", "PALANTIR_JOBS_42", "PALANTIR_JOBS_43", "PALANTIR_JOBS_44", "PALANTIR_JOBS_45", "PALANTIR_JOBS_46", "PALANTIR_JOBS_47", "PALANTIR_JOBS_48", "PALANTIR_JOBS_49", "PALANTIR_JOBS_50", "PALANTIR_JOBS_51", "PALANTIR_JOBS_52", "PALANTIR_JOBS_53", "PALANTIR_JOBS_54", "PALANTIR_JOBS_55", "PALANTIR_JOBS_56", "PALANTIR_JOBS_57", "PALANTIR_JOBS_58", "PALANTIR_JOBS_59", "PALANTIR_JOBS_60", "PALANTIR_JOBS_61", "PALANTIR_JOBS_62", "PALANTIR_JOBS_63", "PALANTIR_JOBS_64", "PALANTIR_JOBS_65", "PALANTIR_JOBS_66", "PALANTIR_JOBS_67", "PALANTIR_JOBS_68", "PALANTIR_JOBS_69", "PALANTIR_JOBS_70", "PALANTIR_JOBS_71", "PALANTIR_JOBS_72", "PALANTIR_JOBS_73", "PALANTIR_JOBS_74", "PALANTIR_JOBS_75", "PALANTIR_JOBS_76", "PALANTIR_JOBS_77", "PALANTIR_JOBS_78", "PALANTIR_JOBS_79", "PALANTIR_JOBS_80", "PALANTIR_JOBS_81", "PALANTIR_JOBS_82", "PALANTIR_JOBS_83", "PALANTIR_JOBS_84"]
 
                }
              ]
            }
            """,
        },

        # Example 3: Philanthropy
        {
            "role": "user",
            "content": "Summarize Palantir's philanthropic initiatives and any major charitable contributions they've made recently.",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "What philanthropic efforts does Palantir have?",
                  "function": "llm_retrieval",
                  "file_names": "PALANTIR_PHILANTHROPY"
                }
              ]
            }
            """,
        },

        # Example 4: Mission statement & London internships
        {
            "role": "user",
            "content": "What is Palantir's mission statement, and do they have any internship roles available in London?",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
              "subquestion_bundle_list": [
                {
                  "question": "What is Palantir's mission statement?",
                  "function": "vector_retrieval",
                  "file_names": "PALANTIR_OVERVIEW"
                },
                {
                  "question": "List Data Science internship roles in London.",
                  "function": "vector_retrieval",
                  "file_names": ["PALANTIR_JOBS_1", "PALANTIR_JOBS_2", "PALANTIR_JOBS_3", "PALANTIR_JOBS_4", "PALANTIR_JOBS_5", "PALANTIR_JOBS_6", "PALANTIR_JOBS_7", "PALANTIR_JOBS_8", "PALANTIR_JOBS_9", "PALANTIR_JOBS_10", "PALANTIR_JOBS_11", "PALANTIR_JOBS_12", "PALANTIR_JOBS_13", "PALANTIR_JOBS_14", "PALANTIR_JOBS_15", "PALANTIR_JOBS_16", "PALANTIR_JOBS_17", "PALANTIR_JOBS_18", "PALANTIR_JOBS_19", "PALANTIR_JOBS_20", "PALANTIR_JOBS_21", "PALANTIR_JOBS_22", "PALANTIR_JOBS_23", "PALANTIR_JOBS_24", "PALANTIR_JOBS_25", "PALANTIR_JOBS_26", "PALANTIR_JOBS_27", "PALANTIR_JOBS_28", "PALANTIR_JOBS_29", "PALANTIR_JOBS_30", "PALANTIR_JOBS_31", "PALANTIR_JOBS_32", "PALANTIR_JOBS_33", "PALANTIR_JOBS_34", "PALANTIR_JOBS_35", "PALANTIR_JOBS_36", "PALANTIR_JOBS_37", "PALANTIR_JOBS_38", "PALANTIR_JOBS_39", "PALANTIR_JOBS_40", "PALANTIR_JOBS_41", "PALANTIR_JOBS_42", "PALANTIR_JOBS_43", "PALANTIR_JOBS_44", "PALANTIR_JOBS_45", "PALANTIR_JOBS_46", "PALANTIR_JOBS_47", "PALANTIR_JOBS_48", "PALANTIR_JOBS_49", "PALANTIR_JOBS_50", "PALANTIR_JOBS_51", "PALANTIR_JOBS_52", "PALANTIR_JOBS_53", "PALANTIR_JOBS_54", "PALANTIR_JOBS_55", "PALANTIR_JOBS_56", "PALANTIR_JOBS_57", "PALANTIR_JOBS_58", "PALANTIR_JOBS_59", "PALANTIR_JOBS_60", "PALANTIR_JOBS_61", "PALANTIR_JOBS_62", "PALANTIR_JOBS_63", "PALANTIR_JOBS_64", "PALANTIR_JOBS_65", "PALANTIR_JOBS_66", "PALANTIR_JOBS_67", "PALANTIR_JOBS_68", "PALANTIR_JOBS_69", "PALANTIR_JOBS_70", "PALANTIR_JOBS_71", "PALANTIR_JOBS_72", "PALANTIR_JOBS_73", "PALANTIR_JOBS_74", "PALANTIR_JOBS_75", "PALANTIR_JOBS_76", "PALANTIR_JOBS_77", "PALANTIR_JOBS_78", "PALANTIR_JOBS_79", "PALANTIR_JOBS_80", "PALANTIR_JOBS_81", "PALANTIR_JOBS_82", "PALANTIR_JOBS_83", "PALANTIR_JOBS_84"]
                }
              ]
            }
            """,
        },
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
    # The LLM should respond with a JSON that matches SubQuestionBundleList
    # e.g.: {"subquestion_bundle_list": [ {...}, {...} ]}
    subquestions_json = json.loads(response.choices[0].message.function_call.arguments)

    # This next line applies Pydantic validation. If the LLM tries
    # to use a file_name that isn't in our enum, a ValidationError is raised.
    subquestions_pydantic_obj = SubQuestionBundleList(**subquestions_json)

    # Finally extract the actual list of subquestions
    subquestions_list = subquestions_pydantic_obj.subquestion_bundle_list

    return subquestions_list, cost

