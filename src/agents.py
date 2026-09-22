from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0
)


# --------------------------------
# DIAGNOSIS AGENT
# --------------------------------

diagnosis_prompt = ChatPromptTemplate.from_template("""
You are the Diagnosis Agent in a field-service troubleshooting system.

Analyze the technician's problem using ONLY the equipment manual.

Identify:
1. The most likely causes
2. The evidence from the manual supporting those causes

Do not invent information.

Equipment manual:
{context}

Technician problem:
{question}

Return a concise diagnosis.
""")


def diagnosis_agent(context, question):

    chain = diagnosis_prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


# --------------------------------
# PARTS AGENT
# --------------------------------

parts_prompt = ChatPromptTemplate.from_template("""
You are the Parts Recommendation Agent in a field-service system.

Based ONLY on the equipment manual and the diagnosis below,
identify the relevant service parts.

Do not recommend parts that are not mentioned in the manual.

Equipment manual:
{context}

Diagnosis:
{diagnosis}

Return:
- Part name
- Part ID if available
- Why the part may be relevant
""")


def parts_agent(context, diagnosis):

    chain = parts_prompt | llm

    response = chain.invoke({
        "context": context,
        "diagnosis": diagnosis
    })

    return response.content


# --------------------------------
# NEXT-BEST-ACTION AGENT
# --------------------------------

action_prompt = ChatPromptTemplate.from_template("""
You are the Next-Best-Action Agent in a field-service
troubleshooting system.

Determine the safest troubleshooting sequence using ONLY
the equipment manual, diagnosis, and parts information.

Do not invent repair procedures.

Equipment manual:
{context}

Diagnosis:
{diagnosis}

Relevant parts:
{parts}

Return a numbered list of recommended actions.

Include safety precautions from the manual when relevant.
""")


def action_agent(context, diagnosis, parts):

    chain = action_prompt | llm

    response = chain.invoke({
        "context": context,
        "diagnosis": diagnosis,
        "parts": parts
    })

    return response.content