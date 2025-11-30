"""
Gradio web interface for Mapping ChatBot.
"""
import os
import time
from pathlib import Path

import gradio as gr

try:
    from src.chatbot import MappingChatBot
except ImportError:
    from chatbot import MappingChatBot

CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1400px !important;
}
footer {
    display: none !important;
}
#chatbot {
    height: 600px;
}
"""


def create_chatbot_interface(root_folder: str):
    """Create and configure the Gradio interface."""
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

    hierarchy_stats = ""
    if chatbot.hierarchy_engine:
        hierarchy_stats = f" | {len(chatbot.hierarchy_engine.domains)} domains, {len(chatbot.hierarchy_engine.operators)} operators"

    def respond(message, history):
        response = chatbot.process_query(message)
        return response

    with gr.Blocks(title="Mapping ChatBot") as demo:
        demo.theme = gr.themes.Soft()
        demo.css = CUSTOM_CSS

        gr.Markdown(
            f"""
            # 🔍 RAFM Chatbot
            ### Ask questions about field mappings in natural language
            
            **Status:** Loaded **{len(chatbot.mappings_data)}** vendors in **{init_time:.2f}s**{hierarchy_stats}
            
            **Hierarchy:** Domain → Module → Source → Vendor → Operator (all UPPERCASE)
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
                placeholder="Ask about field mappings or type /help for commands...",
                show_label=False,
                scale=9,
                container=False,
                lines=1
            )
            submit = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            clear = gr.Button("Clear Chat", scale=1)

        with gr.Accordion("📝 Example Queries - Field Mappings", open=False):
            gr.Examples(
                examples=[
                    "Give me the mapping for field 'customer_id'",
                    "Show mapping for AccountNumber from domain RA",
                    "What is the mapping for 'email' vendor Oracle",
                    "Find all mappings in module CRM",
                    "Show dimension Sales field Revenue",
                    "get me logics for 'event_type' where domain is RA, module is UC and source is MSC and vendor is Nokia",
                    "get me logics for 'event_type' where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU",
                ],
                inputs=msg,
                label="Click any example to try it"
            )

        with gr.Accordion("🗂️ Example Queries - Hierarchy Navigation", open=False):
            gr.Examples(
                examples=[
                    "How many modules under domain RA?",
                    "List vendors under domain RA",
                    "Get number of operators under domain RA",
                    "Vendors under domain RA with more than 2 operators",
                    "Sources under domain RA containing MSC",
                    "How many vendors under module UC?",
                    "List operators under module PI",
                    "Vendors under module UC with more than 3 operators",
                    "Number of vendors under source MSC",
                    "List operators under source HLR",
                    "How many operators under vendor Nokia?",
                    "Operators under vendor Ericsson matching pattern Air*",
                    "Total number of domains",
                    "Total number of operators",
                    "Top 5 vendors with most operators",
                    "Top 10 domains with most modules",
                    "Modules grouped by domain",
                    "Operators grouped by vendor",
                    "Vendors with zero operators",
                    "All unique operator names",
                ],
                inputs=msg,
                label="Click any example to try it"
            )

        with gr.Accordion("⚡ Special Commands", open=False):
            gr.Examples(
                examples=[
                    "/help",
                    "/list",
                    "/stats",
                    "/hierarchy",
                    "/vendors",
                    "/operators",
                    "/modules",
                    "/sources",
                    "/examples"
                ],
                inputs=msg,
                label="Click any command to execute"
            )

        with gr.Accordion("📖 Quick Reference", open=False):
            gr.Markdown("""
            ### Hierarchy Structure
            **Domain → Module → Source → Vendor → Operator**
            
            All values are displayed in UPPERCASE.
            
            ### Special Commands (start with /)
            | Command | Description |
            |---------|-------------|
            | `/help` | Show help guide |
            | `/list` | List all domains with hierarchy |
            | `/stats` | Show statistics |
            | `/hierarchy` | Hierarchy navigation help |
            | `/vendors` | List all vendors |
            | `/operators` | List all operators |
            | `/modules` | List all modules |
            | `/sources` | List all sources |
            | `/examples` | Show example queries |
            
            ### Field Mapping Search Tips
            - Use quotes for exact field names: `'customer_id'`
            - All filters are optional and case-insensitive
            - Combine filters: `field 'email' vendor Oracle module CRM operator Airtel`
            
            ### Hierarchy Navigation Queries
            - **Counts:** "How many modules under domain RA?"
            - **Lists:** "List vendors under module UC"
            - **Filters:** "Vendors with more than 3 operators under domain RA"
            - **Patterns:** "Operators matching pattern 'Air*'"
            - **Global:** "Top 5 vendors by operator count"
            """)

        def user_message(message, history):
            if not history:
                history = []
            new_history = history + [{"role": "user", "content": message}]
            return "", new_history

        def bot_response(history):
            if not history:
                return []
            user_msg = history[-1]["content"] if history[-1]["role"] == "user" else ""
            bot_msg = respond(user_msg, history)
            history.append({"role": "assistant", "content": bot_msg})
            return history

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

        clear.click(lambda: [], None, chatbot_ui, queue=False)

        gr.Markdown("---")
        gr.Markdown("*💡 Tip: Type `/help` for detailed usage instructions or `/examples` for sample queries*")

    return demo


def main():
    ROOT_FOLDER = os.getenv(
        "MAPPING_ROOT_FOLDER",
        r"C:\Users\aditya.prasad\OneDrive - Mobileum\Documents\OneDrive - Mobileum\Tejas N's files - Templates"
    )

    if not Path(ROOT_FOLDER).exists():
        print(f"\nERROR: Root folder not found: {ROOT_FOLDER}")
        print("Please set MAPPING_ROOT_FOLDER environment variable or update ROOT_FOLDER in src/app.py")
        return

    demo = create_chatbot_interface(ROOT_FOLDER)

    print("\n" + "="*70)
    print("* Starting Gradio Web Interface...")
    print("="*70)

    demo.launch(
        server_name="127.0.0.1",
        server_port=8948,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()