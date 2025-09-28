import asyncio
import os
from dataclasses import dataclass
from uuid import uuid4

from agent_framework import (
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatAgent,
    ChatMessage,
    Executor,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.observability import setup_observability
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# USER CONFIGURATION - SET THESE AS ENVIRONMENT VARIABLES
# =============================================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")

# =============================================================================


@dataclass
class StudentResponse:
    messages: list[ChatMessage]


def create_openai_chat_client():
    """Create OpenAI chat client with explicit settings."""

    return AzureOpenAIChatClient(
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=MODEL_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
    )


class StudentAgentExecutor(Executor):
    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id="student"):
        super().__init__(agent=agent, id=id)

    @handler
    async def handle_teacher_question(
        self, messages: list[ChatMessage], ctx: WorkflowContext[StudentResponse]
    ) -> None:
        if messages and "completed" in messages[-1].contents[-1].text.lower():
            await ctx.yield_output(
                "🎉 Student-teacher conversation completed after 2 turns!"
            )
            return

        response = await self.agent.run(messages)
        print(f"Student: {response.messages[-1].contents[-1].text}")

        for message in response.messages:
            if message.role == Role.ASSISTANT:
                await ctx.add_event(
                    AgentRunUpdateEvent(
                        self.id,
                        data=AgentRunResponseUpdate(
                            contents=[TextContent(text=f"Student: {message.contents[-1].text}")],
                            role=Role.ASSISTANT,
                            response_id=str(uuid4()),
                        ),
                    )
                )

        messages.extend(response.messages)
        await ctx.send_message(StudentResponse(messages=messages))


class TeacherAgentExecutor(Executor):
    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id="teacher"):
        super().__init__(agent=agent, id=id)

    async def _handle_response(self, messages, ctx, response):
        print(f"Teacher: {response.messages[-1].contents[-1].text}")
        for message in response.messages:
            if message.role == Role.ASSISTANT:
                await ctx.add_event(
                    AgentRunUpdateEvent(
                        self.id,
                        data=AgentRunResponseUpdate(
                            contents=[TextContent(text=f"Teacher: {message.contents[-1].text}")],
                            role=Role.ASSISTANT,
                            response_id=str(uuid4()),
                        ),
                    )
                )
        messages.extend(response.messages)
        await ctx.send_message(messages)

    @handler
    async def handle_user_message(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        response = await self.agent.run(messages)
        await self._handle_response(messages, ctx, response)

    @handler
    async def handle_student_answer(
        self, student_response: StudentResponse, ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        messages = student_response.messages
        response = await self.agent.run(messages)
        await self._handle_response(messages, ctx, response)


def create_workflow_from_client():
    """Create workflow using OpenAI chat client with explicit settings."""

    # Create OpenAI chat client
    chat_client = create_openai_chat_client()

    # Create student agent
    student_agent = chat_client.create_agent(
        instructions="You are Jamie, a student. Answer teacher questions briefly (1-2 sentences). Don't ask questions back."
    )

    # Create teacher agent
    teacher_agent = chat_client.create_agent(
        instructions="You are Dr. Smith, a teacher. Ask simple questions on different topics without numbering or formatting. Just ask the question directly. After 2 question-answer exchanges, respond with only 'Completed'. Keep questions short and clear."
    )

    # Create executors
    student_executor = StudentAgentExecutor(student_agent)
    teacher_executor = TeacherAgentExecutor(teacher_agent)

    workflow = (
        WorkflowBuilder()
        .add_edge(teacher_executor, student_executor)
        .add_edge(student_executor, teacher_executor)
        .set_start_executor(teacher_executor)
        .build()
    )

    return workflow


async def main():
    """Main function to run the student-teacher workflow."""

    # Configure observability for workflow visualization
    setup_observability(vs_code_extension_port=4317)

    try:
        workflow = create_workflow_from_client()
        message = ChatMessage(
            role=Role.USER, contents=[TextContent("Start the quiz session.")]
        )
        async for _ in workflow.run_stream([message]):
            pass

    except Exception as e:
        print(f"Error running workflow: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
