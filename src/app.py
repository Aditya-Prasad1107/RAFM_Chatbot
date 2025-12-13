"""
Gradio web interface for Mapping ChatBot with Layout viewing support.
Compatible with multiple Gradio versions.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import gradio as gr
from src.chatbot import MappingChatBot
from src.layout_extractor import read_excel_content, format_layout_content_text


def create_chatbot_interface(root_folder: str):
    """
    Create and configure the Gradio interface.
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

    # Create Gradio interface
    with gr.Blocks() as demo:

        # State for layout context
        layout_state = gr.State({
            'active': False,
            'domain': None,
            'module': None,
            'source': None,
            'vendor': None,
            'operators': {},
            'current_file_path': None
        })

        gr.Markdown(
            f"""
            # RAFM Chatbot
            ### Ask questions about field mappings or view layout files

            **Status:** Loaded **{len(chatbot.mappings_data)}** vendors in **{init_time:.2f}s**
            """
        )

        # Main chat interface
        chatbot_ui = gr.Chatbot(
            label="Chat",
            height=500,
            show_label=False
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask about mappings or layouts... (e.g., 'show layout for domain RA, module UC, source MSC, vendor Nokia')",
                show_label=False,
                lines=1
            )
            submit = gr.Button("Send")

        with gr.Row():
            clear = gr.Button("Clear Chat")

        # Layout Panel - using Accordion that starts visible but closed
        with gr.Accordion("Layout Viewer", open=False) as layout_panel:
            layout_info = gr.Markdown("*Query a layout to see available operators*")

            # Use Dropdown instead of Radio for better compatibility
            operator_dropdown = gr.Dropdown(
                choices=[],
                label="Select Operator",
                interactive=True,
                allow_custom_value=False
            )

            with gr.Row():
                view_layout_btn = gr.Button("View Layout")
                download_btn = gr.Button("Download Original File")

            # Scrollable content display
            layout_content = gr.Textbox(
                label="Layout Content (select all and copy with Ctrl+C)",
                lines=20,
                max_lines=40,
                interactive=False
            )

            # File output for download
            download_file = gr.File(label="Click to download")

        with gr.Accordion("Example Queries", open=False):
            gr.Examples(
                examples=[
                    "Give me the mapping for field 'customer_id'",
                    "Show mapping for AccountNumber from source MSC",
                    "What is the mapping for 'email' vendor Oracle",
                    "Find all mappings in module CRM",
                    "get me logics for 'event_type' where domain is RA, module is UC and source is MSC and vendor is Nokia",
                    "show layout for domain RA, module UC, source MSC, vendor Nokia",
                    "list",
                    "stats",
                    "help"
                ],
                inputs=msg,
                label="Click any example to try it"
            )

        with gr.Accordion("Quick Reference", open=False):
            gr.Markdown("""
            ### Commands:
            - `help` - Detailed help guide
            - `list` or `sources` - View all available domains
            - `stats` - View loading statistics

            ### Mapping Queries:
            - Use quotes for exact field names: `'customer_id'`
            - Combine filters: `field 'email' vendor Oracle module CRM`

            ### Layout Queries:
            - `show layout for domain X, module Y, source Z, vendor W`
            """)

        # ==================== Event Handlers ====================

        def process_message(message, history, state):
            """Process user message and update UI accordingly."""
            if not message or not message.strip():
                return "", history or [], state, gr.update(), "", "", None

            if not history:
                history = []

            # Add user message
            history = history + [{"role": "user", "content": message}]

            # Check if layout query
            if chatbot.is_layout_query(message):
                response, metadata = chatbot.process_layout_query(message)

                if metadata and metadata['type'] == 'layout_operators' and metadata['result']['success']:
                    result = metadata['result']

                    # Convert Path objects to strings
                    operators_str = {k: str(v) for k, v in result['operators'].items()}
                    operators_list = list(operators_str.keys())

                    state = {
                        'active': True,
                        'domain': result['domain'],
                        'module': result['module'],
                        'source': result['source'],
                        'vendor': result['vendor'],
                        'operators': operators_str,
                        'current_file_path': None
                    }

                    history.append({"role": "assistant", "content": response})

                    # Return with updated dropdown
                    return (
                        "",
                        history,
                        state,
                        gr.Dropdown(choices=operators_list, value=operators_list[0] if operators_list else None),
                        f"**{result['domain']}/{result['module']}/{result['source']}/{result['vendor']}** - Select operator and click 'View Layout'",
                        "",
                        None
                    )

                elif metadata and metadata['type'] == 'layout_content' and metadata['result']['success']:
                    result = metadata['result']
                    content = result['content']
                    text_content = format_layout_content_text(content)
                    file_path_str = str(result['file_path'])

                    state = {
                        'active': True,
                        'domain': result['domain'],
                        'module': result['module'],
                        'source': result['source'],
                        'vendor': result['vendor'],
                        'operators': {},
                        'current_file_path': file_path_str
                    }

                    history.append({"role": "assistant", "content": response})

                    return (
                        "",
                        history,
                        state,
                        gr.Dropdown(choices=[], value=None),
                        f"**{result['operator']}** - {result['domain']}/{result['module']}/{result['source']}/{result['vendor']}",
                        text_content,
                        file_path_str
                    )
                else:
                    history.append({"role": "assistant", "content": response})
                    return (
                        "",
                        history,
                        state,
                        gr.Dropdown(choices=[], value=None),
                        "*Query a layout to see available operators*",
                        "",
                        None
                    )
            else:
                # Regular mapping query
                response = chatbot.process_query(message)
                history.append({"role": "assistant", "content": response})

                return (
                    "",
                    history,
                    state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update()
                )

        def view_layout(operator, state):
            """View layout content for selected operator."""
            if not state or not state.get('active'):
                return "Please query a layout first (e.g., 'show layout for domain RA, module UC, source MSC, vendor Nokia')", None, state

            if not operator:
                return "Please select an operator from the dropdown.", None, state

            operators = state.get('operators', {})
            if operator not in operators:
                return f"Operator '{operator}' not found.", None, state

            file_path = operators[operator]

            try:
                content = read_excel_content(Path(file_path))
                text_content = format_layout_content_text(content)

                state = state.copy()
                state['current_file_path'] = file_path

                return text_content, file_path, state
            except Exception as e:
                return f"Error reading file: {str(e)}", None, state

        def download_original(state):
            """Return the original file path for download."""
            if not state:
                return None
            file_path = state.get('current_file_path')
            if file_path and Path(file_path).exists():
                return file_path
            return None

        def clear_chat(state):
            """Clear chat and reset state."""
            new_state = {
                'active': False,
                'domain': None,
                'module': None,
                'source': None,
                'vendor': None,
                'operators': {},
                'current_file_path': None
            }
            return [], new_state, gr.Dropdown(choices=[], value=None), "*Query a layout to see available operators*", "", None

        # Wire up events
        msg.submit(
            process_message,
            [msg, chatbot_ui, layout_state],
            [msg, chatbot_ui, layout_state, operator_dropdown, layout_info, layout_content, download_file],
            queue=False
        )

        submit.click(
            process_message,
            [msg, chatbot_ui, layout_state],
            [msg, chatbot_ui, layout_state, operator_dropdown, layout_info, layout_content, download_file],
            queue=False
        )

        view_layout_btn.click(
            view_layout,
            [operator_dropdown, layout_state],
            [layout_content, download_file, layout_state],
            queue=False
        )

        download_btn.click(
            download_original,
            [layout_state],
            [download_file],
            queue=False
        )

        clear.click(
            clear_chat,
            [layout_state],
            [chatbot_ui, layout_state, operator_dropdown, layout_info, layout_content, download_file],
            queue=False
        )

        gr.Markdown("---")
        gr.Markdown("*Tip: Type 'help' for detailed usage instructions*")

    return demo


def main():
    """Main entry point for the application."""

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
        server_port=8848,
        share=False,
        show_error=True,
        quiet=False,
        allowed_paths=[ROOT_FOLDER]
    )


if __name__ == "__main__":
    main()
