import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_mcp_agent_tools():
    """Connects to the server and returns LangChain-compatible tools."""
    from langchain_mcp_adapters.tools import load_mcp_tools
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "quantintel.mcp_servers.agent_swarm_server"]
    )
    
    # We yield the tools to manage the connection properly, 
    # but for simplicity in LangChain, we return a fully connected session.
    # Note: production systems should wrap the lifecycle in a class.
    transport = stdio_client(server_params)
    read, write = await transport.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    
    langchain_tools = await load_mcp_tools(session)
    return session, langchain_tools

async def get_mcp_resource(session: ClientSession, uri: str) -> str:
    res = await session.read_resource(uri)
    return res.contents[0].text if res.contents else ""
