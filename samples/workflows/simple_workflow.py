# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.workflow import (
    Executor,
    WorkflowBuilder,
    WorkflowCompletedEvent,
    WorkflowContext,
    handler,
)
from agent_framework_foundry import FoundryChatClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.attributes import service_attributes
from opentelemetry.trace import set_tracer_provider

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:
    OTLPSpanExporter = None


# Load settings from environment variables
otlp_endpoint = f"http://localhost:{os.getenv('FOUNDRY_OTLP_PORT', '4317')}"


# Configure tracing to capture telemetry spans for visualization.
def set_up_tracing():
    if otlp_endpoint and OTLPSpanExporter:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        resource = Resource.create(
            {service_attributes.SERVICE_NAME: "StudentTeacherWorkflow"}
        )
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        set_tracer_provider(tracer_provider)


class StudentAgentExecutor(Executor):
    """
    StudentAgentExecutor

    Executor that handles a "teacher question" event by re-invoking the agent with
    the current conversation messages and requesting a response.

    Parameters (for the handler):
    - response: AgentExecutorResponse containing the prior agent run result and messages.
    - ctx: WorkflowContext[None] used to carry workflow-level state, cancellation, or metadata.
    """

    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id="student"):
        super().__init__(agent=agent, id=id)

    @handler
    async def handle_teacher_question(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        # wait 2 seconds to simulate "thinking"
        await asyncio.sleep(2)

        response = await self.agent.run(messages)
        # Extract just the text content from the last message
        print(f"Student: {response.messages[-1].contents[-1].text}")

        messages.extend(response.messages)
        await ctx.send_message(messages)


class TeacherAgentExecutor(Executor):
    """
    TeacherAgentExecutor

    Orchestrates the "teacher" side of the student-teacher workflow.

    - Start the conversation by sending the initial teacher prompt to the agent.
    - Receive the student's responses, track the number of turns, and decide when to
      end the workflow (either after a configured number of turns or when a completion
      token is observed).
    - Re-invoke the teacher agent to ask the next question when appropriate.
    """

    turn_count: int = 0
    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id="teacher"):
        super().__init__(agent=agent, id=id, turn_count=0)

    @handler
    async def handle_start_message(
        self, message: str, ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        """
        Handle the initial start message for the teacher.

        The incoming message is treated as a user chat message sent to the teacher agent.
        We wrap it in a ChatMessage and create an AgentExecutorRequest asking the agent
        to respond.
        """
        # Build a user message for the teacher agent and request a response
        chat_message = ChatMessage(Role.USER, text=message)
        messages: list[ChatMessage] = [chat_message]
        response = await self.agent.run(messages)
        # Extract just the text content from the last message
        print(f"Teacher: {response.messages[-1].contents[-1].text}")

        messages.extend(response.messages)
        await ctx.send_message(messages)

    @handler
    async def handle_student_answer(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        """
        Handle the student's answer (a list of ChatMessages).

        Behavior:
        - Increment the turn counter each time the teacher processes a student's answer.
        - If the turn limit is reached, emit a WorkflowCompletedEvent to end the workflow.
        - Otherwise, forward the conversation messages back to the teacher agent and request
          the next question.
        """
        # wait 2 seconds to simulate "thinking"
        await asyncio.sleep(2)
        self.turn_count += 1

        # End after 5 turns to avoid infinite conversation loops
        if self.turn_count >= 5:
            await ctx.add_event(WorkflowCompletedEvent())
            return

        # Otherwise, ask the teacher agent to produce the next question using the current messages
        response = await self.agent.run(messages)
        print(f"Teacher: {response.messages[-1].contents[-1].text}")

        messages.extend(response.messages)
        await ctx.send_message(messages)


async def main():
    set_up_tracing()

    credential = AzureCliCredential()
    client = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
    )

    # Create the Student and Teacher agents
    student_agent = await client.agents.create_agent(
        model=os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"],
        name="StudentAgent",
        instructions="""You are Jamie, a student. Your role is to answer the teacher's questions briefly and clearly.

            IMPORTANT RULES:
            1. Answer questions directly and concisely
            2. Keep responses short (1-2 sentences maximum)
            3. Do NOT ask questions back""",
    )
    teacher_agent = await client.agents.create_agent(
        model=os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"],
        name="TeacherAgent",
        instructions="""You are Dr. Smith, a teacher. Your role is to ask the student different, simple questions to test their knowledge.

            IMPORTANT RULES:
            1. Ask ONE simple question at a time
            2. NEVER repeat the same question twice
            3. Ask DIFFERENT topics each time (science, math, history, geography, etc.)
            4. Keep questions short and clear
            5. Do NOT provide explanations - only ask questions""",
    )

    try:
        student_executor = StudentAgentExecutor(
            ChatAgent(
                chat_client=FoundryChatClient(client=client, agent_id=student_agent.id)
            ),
        )

        teacher_executor = TeacherAgentExecutor(
            ChatAgent(
                chat_client=FoundryChatClient(client=client, agent_id=teacher_agent.id)
            ),
        )

        # Define the workflow orchestration
        workflow = (
            WorkflowBuilder()
            .add_edge(teacher_executor, student_executor)
            .add_edge(student_executor, teacher_executor)
            .set_start_executor(teacher_executor)
            .build()
        )

        async for event in workflow.run_stream("Start the quiz session."):
            if isinstance(event, WorkflowCompletedEvent):
                print(f"\n🎉 Student-teacher conversation completed after 5 turns!")

    except Exception as e:
        print(f"Error running workflow: {e}")
    finally:
        try:
            # Clean up the agents
            if student_agent:
                await client.agents.delete_agent(student_agent.id)
            if teacher_agent:
                await client.agents.delete_agent(teacher_agent.id)
            await client.close()
            await credential.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
