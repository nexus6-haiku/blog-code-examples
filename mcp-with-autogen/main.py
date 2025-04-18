#!/usr/bin/env python3
# python3 -m pip install -U autogen-ext[mcp] json-schema-to-pydantic>=0.2.2 anthropic

import asyncio
import argparse
import sys
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_core import CancellationToken
from autogen_agentchat.ui import Console

async def main(task: str) -> None:
    # Setup server params for fetching remote content
    fetch_mcp_server = StdioServerParams(command="python3", args=["-m", "mcp_server_fetch"])
    fetch_tools = await mcp_server_tools(fetch_mcp_server)
    
    # Setup server params for local filesystem access with correct path
    fs_mcp_server = StdioServerParams(
        command="npx", 
        args=["-y", "@modelcontextprotocol/server-filesystem", "/Dati/workspace/Python/mcp-agent/example"]
    )
    fs_tools = await mcp_server_tools(fs_mcp_server)
    
    # Combine all tools
    all_tools = fetch_tools + fs_tools
    
    # Create an agent that uses Anthropic Claude with detailed system prompt
    model_client = AnthropicChatCompletionClient(
        model="claude-3-7-sonnet-20250219",
        system_prompt=(
            "You are a helpful AI assistant with access to tools for fetching content and "
            "interacting with the filesystem. When asked to create projects or code, "
            "make sure to create comprehensive implementations with all necessary files. "
            "For programming tasks, include all required files, not just headers or minimal "
            "implementations. For Haiku development specifically:"
            "\n\n1. Create both header (.h) AND implementation (.cpp) files for ALL classes"
            "\n2. Include a complete Makefile"
            "\n3. Create a main.cpp file if appropriate"
            "\n4. Implement ALL necessary methods in the .cpp files, not just declarations"
            "\n5. Create a full working project structure"
            "\n6. Implement full menu functionality including About windows"
            "\n7. Make sure to handle all event handling and lifecycle methods"
            "\n\nDo not stop at minimal implementations. Create complete, working code."
        )
    )
    
    # Create the agent with reflection DISABLED
    agent = AssistantAgent(
        name="assistant", 
        model_client=model_client, 
        tools=all_tools,
        # Explicitly disable reflection to avoid the error
        reflect_on_tool_use=False
    )
    
    # Increase the message limit for complex tasks
    termination = MaxMessageTermination(
        max_messages=20) | TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([agent], termination_condition=termination)
    
    await Console(team.run_stream(
        task=task, 
        cancellation_token=CancellationToken()
    ))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run an AI agent to process a task or URL")
    parser.add_argument("task", nargs="?", default=None, help="The task or URL to process")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no task is provided, display help and exit
    if not args.task:
        parser.print_help()
        print("\nError: You must provide a task or URL")
        sys.exit(1)
    
    # Run the main function with the provided task
    asyncio.run(main(args.task))
