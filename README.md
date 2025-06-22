@cl.on_message
async def main(message: cl.Message):
    response = f'Received message: {message.content}'
    await cl.Message(content=response).send()