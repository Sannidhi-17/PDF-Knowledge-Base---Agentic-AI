from agno.agent import Agent
from agno.models.groq import Groq
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

embedder = HuggingfaceCustomEmbedder(id="sentence-transformers/all-MiniLM-L6-v2")

knowledge = Knowledge(
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="recipes",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    description="You are a Thai Cuisine Expert!",
    instructions=[
        "Search your knowledge base for thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results.",
    ],
    knowledge=knowledge,
    tools=[DuckDuckGoTools()],
    # show_tool_calls=True,
    markdown=True,
)

# Comment out after the knowledge base is loaded
knowledge.add_content(
    url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
    reader=PDFReader(),
)

agent.print_response("How many recipes are available?", stream=True)
agent.print_response("What is the history of Thai Curry?", stream=True)