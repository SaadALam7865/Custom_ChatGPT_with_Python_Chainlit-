import chainlit as cl
import os
from dotenv import load_dotenv
from typing import cast
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Initialize the Chainlit app
@cl.on_chat_start
async def on_chat_start():
    # Initialize the OpenAI client with the API key
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url='https://generativelanguage.googleapis.com/v1beta/openai'
    )

    # Initialize the OpenAIChatCompletionsModel with the external client
    model = OpenAIChatCompletionsModel(
        model='gemini-2.0-flash',
        openai_client=external_client,
    )

    # Create a RunConfig with the model
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True,
    )
    # user session to store chat history...
    cl.user_session.set('chat history',[])
    cl.user_session.set('config',config)

    # Create an agent with the model and config
    agent:Agent = Agent(
        name='Assistant',
        instructions='You are a helpful assistant.',
        model=model
    )
    cl.user_session.set('agent',agent)
    await cl.Message(content='Welcome to the Assistant! How can I help you today?').send()

@cl.on_message # handle received mesages from the user
async def main(message: cl.Message):
    msg = cl.Message(content='Processing your request...')
    await msg.send()

    # Retrieve the agent and config from user session
    agent:Agent = cast(Agent, cl.user_session.get('agent'))
    config:RunConfig = cast(RunConfig, cl.user_session.get('config'))
    # Create a history object to store chat history
    history = cl.user_session.get('chat history') or []
    history.append({'role': 'user', 'content':message.content})
    try:
        print("\n[CALLING_ AGENT_WITH_CONTEXT] \n", history, "\n") 
        result = Runner.run_sync(
            starting_agent=agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        # user and assistant messages to new chat history
        cl.user_session.set('chat history', result.to_input_list())

        # both messages to chat history
        print(f'User Message: {message.content}')
        print(f'Assistant Message: {response_content}')

    except Exception as e:
        msg.content = f'An error occurred: {str(e)}'
        await msg.update()
        print(f'Error: {str(e)}')
        




    
