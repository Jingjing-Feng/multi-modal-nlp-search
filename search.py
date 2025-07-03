import os
from typing import List, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import rich

from model.sbert import SBertModel

from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


ES = Elasticsearch("http://localhost:9200/")
INDEX_NAME = "nls"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class File:
    filename: str
    extension: str
    path: str
    created: datetime
    size: int


@dataclass
class NLSResults:
    """
    A class to hold the results of a natrual language search results.
    """

    result_type: str  # "search" or "answer"
    files: list[str]  # list of files that match the query
    answer: str


FILE_CACHE: dict[str, File] = {}


def _extract_and_cache_file(hits: list[Any]) -> list[str]:
    files: list[str] = []
    for hit in hits:
        FILE_CACHE[hit["_source"]["metadata"]["filename"]] = File(
            filename=hit["_source"]["metadata"]["filename"],
            extension=hit["_source"]["metadata"]["extension"],
            path=hit["_source"]["metadata"]["path"],
            created=hit["_source"]["metadata"]["created"],
            size=hit["_source"]["metadata"]["size"],
        )

        files.append(hit["_source"]["metadata"]["filename"])
    return files


@tool
def semantic_search(query: str) -> NLSResults:
    """
    Return the files that match the query using both semantic and keyword search.

    Parameters
    ----------
    query : str
        The query to search for.

    Returns
    -------
    NLSResults
        The results of the search.

    Here is the definition of the NLSResults class:
    @dataclass
    class NLSResults:
        result_type: str  # "search" or "answer"
        file: list[str]  # list of files that match the query
        answer: str
    """
    query_embedding = SBertModel().get_embedding(query)
    resp = ES.search(
        index=INDEX_NAME,
        query={
            # "bool": {
            #     "should": [
            #         {"match": {"filename": {"query": query, "fuzziness": "AUTO"}}},
            #         {"match": {"text": {"query": query, "fuzziness": "AUTO"}}},
            #     ],
            # }
            "multi_match": {
                "query": query,
                "type": "most_fields",
                "fields": ["filename", "text"],
                "fuzziness": "AUTO",
            }
        },
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 1,
            "num_candidates": 3,
        },
    )

    # Extract the files from the response and cache them
    files = _extract_and_cache_file(resp["hits"]["hits"])

    return NLSResults(result_type="search", files=files, answer="")


@tool
def get_time_range_file(start_time: datetime, end_time: datetime) -> NLSResults:
    """
    Return the files that were created in the given time range.

    Parameters
    ----------
    start_time : datetime
        The start time of the range.
    end_time : datetime
        The end time of the range.

    Returns
    -------
    NLSResults
        The results of the search.

    Here is the definition of the NLSResults class:
    @dataclass
    class NLSResults:
        result_type: str  # "search" or "answer"
        file: list[str]  # list of files that match the query
        answer: str
    """
    resp = ES.search(
        index=INDEX_NAME,
        query={
            "range": {
                "created": {
                    "gte": start_time,
                    "lte": end_time,
                }
            }
        },
    )

    files = _extract_and_cache_file(resp["hits"]["hits"])
    return NLSResults(result_type="search", files=files, answer="")


@tool
def get_extension_file(extension: str) -> NLSResults:
    """
    Return the files that have the given extension.

    Parameters
    ----------
    extension : str
        The extension to search for.

    Returns
    -------
    NLSResults
        The results of the search.

    Here is the definition of the NLSResults class:
    @dataclass
    class NLSResults:
        result_type: str  # "search" or "answer"
        file: list[str] # list of files that match the query
        answer: str
    """
    resp = ES.search(
        index=INDEX_NAME,
        query={
            "match": {
                "extension": extension,
            }
        },
    )
    files = _extract_and_cache_file(resp["hits"]["hits"])
    return NLSResults(result_type="search", files=files, answer="")


@tool
def aggregate_results(results: List[NLSResults]) -> NLSResults:
    """
    Use this tool by passing a dictionary: {"results": [NLSResults1, NLSResults2, ...]}.
    This function returns the files common to all NLSResults entries.

    Parameters
    ----------
    results : list[NLSResults]
        The results to aggregate.

    Returns
    -------
    NLSResults
        The aggregated results.

    Here is the definition of the NLSResults class:
    @dataclass
    class NLSResults:
        result_type: str  # "search" or "answer"
        file: list[str]  # list of files that match the query
        answer: str
    """
    files = set(results[0].files)
    for result in results[1:]:
        files &= set(result.files)
    files = list(files)

    return NLSResults(result_type="search", files=files, answer="")


def _es_query(query: str):
    return {
        "query": {
            "multi_match": {
                "query": query,
                "type": "most_fields",
                "fields": ["filename", "text"],
                "fuzziness": "AUTO",
            },
        },
        "knn": {
            "field": "embedding",
            "query_vector": SBertModel.get_embedding(query),
            "k": 1,
            "num_candidates": 3,
        },
    }


@tool
def question_answering(question: str) -> NLSResults:
    """
    Perform question answering on the files.

    Parameters
    ----------
    question : str
        The question to answer.

    Returns
    -------
    NLSResults
        The results of the question answering.

    Here is the definition of the NLSResults class:
    @dataclass
    class NLSResults:
        result_type: str  # "search" or "answer"
        file: list[str]"
    """

    es_retriever = ElasticsearchRetriever.from_es_params(
        url="http://localhost:9200/",
        index_name=INDEX_NAME,
        content_field="text",
        body_func=_es_query,
    )

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        openai_api_key=openai_api_key,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=es_retriever,
        return_source_documents=True,
    )

    resp = qa.invoke(question)

    answer = resp["result"]
    files: list[str] = []
    for doc in resp["source_documents"]:
        FILE_CACHE[doc.metadata["_source"]["metadata"]["filename"]] = File(
            filename=doc.metadata["_source"]["metadata"]["filename"],
            extension=doc.metadata["_source"]["metadata"]["extension"],
            path=doc.metadata["_source"]["metadata"]["path"],
            created=doc.metadata["_source"]["metadata"]["created"],
            size=doc.metadata["_source"]["metadata"]["size"],
        )

        files.append(doc.metadata["_source"]["metadata"]["filename"])

    return NLSResults(result_type="answer", files=files, answer=answer)


SYSTEM_PROMPT = """ You are a highly capable assistant designed to help with searching for files and answering questions about them. You have access to specialized tools for different types of queries. You always have to use at least one tool.

1. **Question Answering Tool** (`question_answering`):
   Use this tool when the query asks something about the *content of the files*. A query qualifies if it:
   - Ends with a question mark (?)
   - Starts with question words like: what, what's, how, why, when, where, who, which
   - Requests instructions or explanations (e.g., "explain...", "tell me about...", "steps to...")

2. **Time-Ranged Search Tool** (`get_time_range_file`):
   Use this tool when the query involves specific dates or time ranges.
   - You do not need to manually calculate date ranges — let the tool handle that.
   - The current Unix timestamp is {current_time}.

3. **Semantic Search Tool** (`semantic_search`):
   Use this tool for general keyword-based or concept-based searches that do *not* involve specific time ranges or file types.
   - This includes natural language phrases or topics that don't clearly fit into the other categories.

4. **Extension-Based File Search Tool** (`get_extension_file`):
   Use this tool when the query is about finding files of a specific type, such as:
   - PDFs, images, audio files (e.g., ".pdf", ".jpg", ".mp3", etc.)
   - Example queries: "Show me all image files", "Find .mp3 files"

5. **Result Aggregation Tool** (`aggregate_results`):
   Use this tool when a query involves *multiple filters* (e.g., time + keyword, extension + keyword).
   - First, run the individual tools separately.
   - Then, use this tool to get the intersection (common files) between results.
"""


def natural_language_search(
    query: str,
) -> tuple[NLSResults, dict[str, File], List[str]]:
    tools = [
        semantic_search,
        get_time_range_file,
        question_answering,
        get_extension_file,
        aggregate_results,
        PythonREPLTool(),
    ]

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        openai_api_key=openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.format(current_time=datetime.now())),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        verbose=True,
    )

    results = agent_executor.invoke({"input": query})
    print(results)
    
    # Handle case where no search was performed (e.g. greetings)
    if not results["intermediate_steps"]:
        return NLSResults(result_type="message", files=[], answer=results["output"]), {}, []
        
    search_results = results["intermediate_steps"][-1][-1]

    try:
        if search_results.result_type == "search":
            search_results.files = [
                f for f in search_results.files if f in results["output"]
            ]
        elif search_results.result_type not in ["search", "answer", "message"]:
            raise ValueError(f"Invalid result type: {search_results.result_type}")

        file_metas = {
            fname: f for fname, f in FILE_CACHE.items() if fname in search_results.files
        }

        tools_used = [step[0].tool for step in results["intermediate_steps"]]
        return search_results, file_metas, tools_used

    except Exception as e:
        rich.print(e)
    raise e
