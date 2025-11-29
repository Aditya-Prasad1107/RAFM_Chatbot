"""
Gradio web interface for Mapping ChatBot.
"""
import os
import time
from pathlib import Path

import gradio as gr
from src.chatbot import MappingChatBot

# Custom CSS for better UI
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1400px !important;
}

.message {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}

footer {
    display: none !important;
}

#chatbot {
    height: 600px;
}

.contain {
    max-width: 100% !important;
}
"""


def create_chatbot_interface(root_folder: str):
    """
    Create and configure the Gradio interface.

    Args:
        root_folder: Path to the root folder containing the data structure

    Returns:
        Gradio Blocks interface
    """

    # Initialize chatbot
    print("\n" + "="*70)
    print("Initializing Mapping ChatBot...")
    print("="*70)

    start = time.time()
    chatbot = MappingChatBot(
        root_folder,
        use_parallel=True,
        max_workers=8,
        cache_enabled=True,
        cache_dir="./cache",
        cache_size_mb=500,
        cache_ttl_hours=24
    )
    chatbot.load_all_mappings()
    init_time = time.time() - start

    if not chatbot.mappings_data:
        print("\nWARNING: No data loaded! Please check your folder structure.")

    # Chat response function
    def respond(message, history):
        """Generate response for user message."""
        response = chatbot.process_query(message)
        return response

    # Create Gradio interface
    with gr.Blocks(title="Mapping ChatBot") as demo:
        demo.theme = gr.themes.Soft()
        demo.css = CUSTOM_CSS

        gr.Markdown(
            f"""
            # RAFM Chatbot
            ### Ask questions about field mappings in natural language
            
            **Status:** Loaded **{len(chatbot.mappings_data)}** vendors in **{init_time:.2f}s**
            """
        )

        chatbot_ui = gr.Chatbot(
            label="Chat",
            height=600,
            show_label=False,
            elem_id="chatbot"
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask about field mappings... (e.g., 'Show mapping for customer_id')",
                show_label=False,
                scale=9,
                container=False,
                lines=1
            )
            submit = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            clear = gr.Button("Clear Chat", scale=1)

        with gr.Accordion("Example Queries", open=False):
            gr.Examples(
                examples=[
                    "Give me the mapping for field 'customer_id'",
                    "Show mapping for AccountNumber from source SAP",
                    "What is the mapping for 'email' vendor Oracle",
                    "Find all mappings in module CRM",
                    "Show dimension Sales field Revenue",
                    "get me logics for 'event_type' where source is RA, module is UC and source name is MSC and vendor is Nokia",
                    "get me logics for 'event_type' where source is RA, module is UC, source name is MSC, vendor is Nokia and operator is DU",
                    "list",
                    "stats",
                    "cache stats",
                    "clear cache",
                    "help"
                ],
                inputs=msg,
                label="Click any example to try it"
            )

        with gr.Accordion("Quick Reference", open=False):
            gr.Markdown("""
            ### Commands:
            - `help` - Detailed help guide
            - `list` or `sources` - View all available sources
            - `stats` - View loading statistics
            - `cache stats` - View cache performance
            - `clear cache` - Clear all cached files
            
            ### Search Tips:
            - Use quotes for exact field names: `'customer_id'`
            - All filters are optional and case-insensitive
            - Combine filters: `field 'email' vendor Oracle module CRM operator Airtel`
            - Results show the complete drill-down hierarchy including operator and filename
            - **Operator** is extracted from filenames automatically
            - Filter by operator: `... and operator is DU`
            """)

        # Event handlers
        def user_message(message, history):
            """Add user message to chat."""
            if not history:
                history = []
            new_history = history + [{"role": "user", "content": message}]
            return "", new_history

        def bot_response(history):
            """Generate and add bot response."""
            if not history:
                return []
            user_msg = history[-1]["content"] if history[-1]["role"] == "user" else ""
            bot_msg = respond(user_msg, history)
            history.append({"role": "assistant", "content": bot_msg})
            return history

        # Submit on Enter or button click
        msg.submit(
            user_message,
            [msg, chatbot_ui],
            [msg, chatbot_ui],
            queue=False
        ).then(
            bot_response,
            chatbot_ui,
            chatbot_ui
        )

        submit.click(
            user_message,
            [msg, chatbot_ui],
            [msg, chatbot_ui],
            queue=False
        ).then(
            bot_response,
            chatbot_ui,
            chatbot_ui
        )

        # Clear chat
        clear.click(lambda: [], None, chatbot_ui, queue=False)

        gr.Markdown("---")
        gr.Markdown("*Tip: Type 'help' for detailed usage instructions*")

    return demo


def main():
    """Main entry point for the application."""

    # Configuration - Use environment variable or default path
    ROOT_FOLDER = os.getenv(
        "MAPPING_ROOT_FOLDER",
        r"C:\Users\aditya.prasad\OneDrive - Mobileum\Documents\OneDrive - Mobileum\Tejas N's files - Templates"
    )

    # Validate path
    if not Path(ROOT_FOLDER).exists():
        print(f"\nERROR: Root folder not found: {ROOT_FOLDER}")
        print("Please set MAPPING_ROOT_FOLDER environment variable or update ROOT_FOLDER in src/app.py")
        return

    # Create and launch interface
    demo = create_chatbot_interface(ROOT_FOLDER)

    print("\n" + "="*70)
    print("* Starting Gradio Web Interface...")
    print("="*70)

    demo.launch(
        server_name="127.0.0.1",
        server_port=8848,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()