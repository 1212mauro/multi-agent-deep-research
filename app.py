import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from google_search import google_search_tool

try:  # optional: load API keys from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set the GEMINI_API_KEY environment variable (see .env.example).")

    model_client = OpenAIChatCompletionClient(
        model= "gemini-2.5-pro-preview-03-25",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        api_type= "google",
        model_info={
            "family": ModelFamily.GEMINI_2_5_PRO,
            "function_calling": True,
            "json_output": True,
            "vision": False,
            "temperature": 0.2
        }
    )

    research_assistant = AssistantAgent(
        name="research_assistant",
        description="A research assistant that performs web searches and analyzes information",
        model_client=model_client,
        tools=[google_search_tool],
        system_message="""You are a research assistant focused on finding accurate information.
        Use the google_search tool to find relevant information.
        Break down complex queries into specific search terms.
        Always verify information across multiple sources when possible.
        Focus on scientific articles over popular forums.
        When you find relevant information, explain why it's relevant and how it connects to the query. 
        When you get feedback from the a verifier agent, use your tools to act on the feedback and make progress."""
    )

    user_proxy_agent = UserProxyAgent(
        "UserProxyAgent",
        description="A user to approve or disapprove tasks.",
    )

    verifier = AssistantAgent(
        name="verifier",
        description="A verification specialist who ensures research quality and completeness",
        model_client=model_client,
        system_message="""You are a research verification specialist.
        Your role is to:
        1. Verify that search queries are effective and suggest improvements if needed
        2. Explore drill downs where needed e.g, if the answer is likely in a link in the returned search results, suggest clicking on the link
        3. Suggest additional angles or perspectives to explore. Be judicious in suggesting new paths to avoid scope creep or wasting resources, if the task appears to be addressed and we can provide a report, respond with "APPROVED".
        4. Track progress toward answering the original question
        5. When the research is complete, provide a detailed summary in markdown format

        For incomplete research, end your message with "CONTINUE RESEARCH".
        For complete research, end your message with APPROVED.
        never end your message with TERMINATE.

        Your responses should be structured as:
        - Progress Assessment
        - Gaps/Issues (if any)
        - Suggestions (if needed)
        """
    )

    summary_agent = AssistantAgent(
        name="summary_agent",
        description="A summary agent that provides a detailed markdown summary of the research as a report to the user.",
        model_client=model_client,
        system_message="""You are a summary agent. Your role is to provide a detailed markdown summary of the research as a report to the user. Your report should have a reasonable title that matches the research question and should summarize the key details in the results found in natural an actionable manner. The main results/answer should be in the first paragraph.
        The last part of your report should list all the sources that the research_agent visited.
        Your report must end with the word "TERMINATE" to signal the end of the conversation."""
    )

    text_termination = TextMentionTermination("TERMINATE")
    max_messages = MaxMessageTermination(max_messages=30)
    termination = text_termination | max_messages

    selector_prompt = """You are coordinating a research team by selecting the team member to speak/act next. The following team member roles are available:
    {roles}.
    The research_assistant performs searches and analyzes information.
    The user_proxy_agent returns requested input from the user.
    The verifier evaluates progress and ensures completeness.
    The summary_agent provides a detailed markdown summary of the research as a report to the user.

    Given the current context, select the most appropriate next speaker.
    The research_assistant should search and analyze.
    The user_proxy_agent should be used for getting input from the user.
    The verifier should evaluate progress and guide the research (select this role is there is a need to verify/evaluate progress). You should ONLY select the summary_agent role if the research is complete and it is time to generate a report.

    Base your selection on:
    1. Current stage of research
    2. Last speaker's findings or suggestions
    3. Need for verification vs need for new information

    Read the following conversation. Then select the next role from {participants} to play. Only return the role.

    {history}

    Read the above conversation. Then select the next role from {participants} to play. ONLY RETURN THE ROLE."""

    team = SelectorGroupChat(
        participants=[research_assistant, user_proxy_agent, verifier, summary_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True
    )

    task = input("Prompt: ")

    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())