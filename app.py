import chainlit as cl
from chainlit.element import Element
from datetime import datetime

from search import natural_language_search, File


# Does this function need to be async?
def _get_file_elements(file_meta: File) -> Element:
    element_cls_by_ext = {
        ".pdf": cl.Pdf,
        ".txt": cl.Text,
        ".mp3": cl.Audio,
        # ".mp4": cl.Video,
        ".png": cl.Image,
        ".jpg": cl.Image,
        ".jpeg": cl.Image,
    }

    element_cls = element_cls_by_ext.get(file_meta.extension)
    if element_cls is None:
        raise ValueError(f"Unsupported file extension: {file_meta.extension}")
    return element_cls(path=file_meta.path, name=file_meta.filename, display="side")


@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! Please enter the qeury you want to ask.").send()


@cl.on_message
async def main(message: cl.Message):
    async with cl.Step("Processing your query...") as step:
        search_results, file_metas, tools_used = natural_language_search(
            message.content
        )

        # Handle direct messages (e.g. greetings)
        if search_results.result_type == "message":
            await cl.Message(content=search_results.answer).send()
            return

        # If no files matched
        if len(search_results.files) == 0 and search_results.result_type == "search":
            await cl.Message(content="No results found.").send()
            return

        if search_results.result_type == "search":
            message_content = "Here are the search results:\n\n"
        else:
            message_content = f"{search_results.answer}\n\nSources:\n\n"

        # Display the search results of files
        message_content += "| File | Size | Created |\n|------|------|---------|\n"
        for fname, file in file_metas.items():
            size_mb = file.size / (1024 * 1024)
            date = datetime.fromisoformat(file.created).strftime("%Y-%m-%d %H:%M:%S")
            message_content += f"| {fname} | {size_mb} MB | {date} |\n"

        elements = [_get_file_elements(f_m) for _, f_m in file_metas.items()]
        await cl.Message(content=message_content, elements=elements).send()

    step.output = "Tools used: " + ", ".join([f"`{tool}`" for tool in tools_used])
    await step.update()
