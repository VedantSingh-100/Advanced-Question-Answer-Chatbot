from openai_utils import llm_call
def response_aggregator(llm_model, question, responses):
    """Aggregates the responses from the subquestions to generate the final response.
    """
    print("-------> ⭐ Aggregating responses...")
    system_prompt = """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise."""

    context = ""
    for i, response in enumerate(responses):
        context += f"\n{response}"

    user_prompt = f"""Question: {question}
                      Context: {context}
                      Answer:"""

    response, cost = llm_call(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
    answer = response.choices[0].message.content
    # answer = response.generated_text

    return answer, cost