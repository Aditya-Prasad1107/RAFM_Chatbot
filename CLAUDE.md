# RAFM Chatbot - Project Scope

## Overview

**RAFM Chatbot** (Rapid Application Field Mapping Chatbot) is a natural language interface for querying and discovering field mappings stored in Excel files. Users can ask questions in natural language to find mapping information across a hierarchical data structure. It also supports viewing Layout/Format/Decoder files.

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.7+ | Core language |
| Gradio | 6.0.1 | Web UI framework |
| openpyxl | 3.1.2 | Excel file parsing |
| tqdm | 4.66.1 | Progress tracking |
| huggingface-hub | >=0.33.5 | Future ML integration |

## Project Structure

```
RAFM_Chatbot/
├── main.py                 # PyCharm entry point (template)
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # This file - project scope reference
├── src/
│   ├── __init__.py        # Package initializer
│   ├── app.py             # Gradio web interface (entry point)
│   ├── chatbot.py         # Core chatbot logic & NLP processing
│   ├── extractor.py       # Excel field extraction (LdRules)
│   ├── layout_extractor.py # Layout file extraction
│   ├── cache.py           # Caching mechanism for performance
│   └── cache/             # Cache storage directory
└── .gitignore
```

## Core Components

### 1. `src/app.py` - Web Interface
- Creates Gradio Blocks UI on `127.0.0.1:8848`
- Initializes chatbot with parallel loading and caching
- Handles user input and displays formatted responses
- **Layout Panel**: Operator buttons, scrollable content, download/copy features
- Entry point: `python src/app.py`

### 2. `src/chatbot.py` - MappingChatBot Class
- **Query Parsing**: NLP-based entity extraction using regex patterns
- **Search Engine**: Intelligent field matching with confidence scoring
- **Fuzzy Matching**: Exact, prefix, contains, Jaccard similarity, Levenshtein distance
- **Operator Extraction**: Derives operator names from Excel filenames
- **Layout Support**: Detects layout queries and routes to layout handling
- **Special Commands**: `help`, `list`, `stats`, `cache stats`, `clear cache`

### 3. `src/extractor.py` - Excel Extractor (LdRules)
- Parses `.xlsx` files from LdRules folders
- Extracts: Column B (Dimension/Measure), Column C (Field), Column D (Expression)
- Tracks source filename for each mapping
- Filters Excel errors (#REF!, #N/A, etc.)

### 4. `src/layout_extractor.py` - Layout Extractor
- Parses `.xlsx` files from Layout folders
- Extracts operator names from filenames (prefix: `Layout_`)
- Reads all sheets and formats content
- Supports text and markdown table output

### 5. `src/cache.py` - ExcelCache
- Caches Excel extraction results with 24-hour TTL
- MD5 hash-based cache keys
- Modification time and content hash validation
- Size-based cleanup (500 MB default limit)

## Expected Data Structure

```
MAPPING_ROOT_FOLDER/
├── Domain/
│   ├── Module/
│   │   ├── Source/
│   │   │   ├── Vendor/
│   │   │   │   ├── LdRules/
│   │   │   │   │   ├── LdRules_Vendor.xlsx
│   │   │   │   │   └── LdRules_Vendor_Operator.xlsx
│   │   │   │   └── Layout/
│   │   │   │       ├── Layout_Vendor.xlsx
│   │   │   │       └── Layout_Vendor_Operator.xlsx
```

**4-Level Hierarchy:**
1. **Domain** - Top-level (e.g., RA, CRM, ERP)
2. **Module** - Business area (e.g., UC, Billing)
3. **Source** - Source system (e.g., MSC, Oracle)
4. **Vendor** - Provider (e.g., Nokia, Airtel)

**Two Folder Types:**
- `LdRules/` - Contains mapping logic Excel files
- `Layout/` - Contains format/decoder/layout Excel files

## Configuration

### Environment Variables
- `MAPPING_ROOT_FOLDER` - Path to hierarchical data structure (required)

### Chatbot Options (in `app.py`)
```python
MappingChatBot(
    root_folder,
    use_parallel=True,      # Enable parallel processing
    max_workers=8,          # Thread pool size
    cache_enabled=True,     # Enable caching
    cache_dir="./cache",    # Cache location
    cache_size_mb=500,      # Max cache size
    cache_ttl_hours=24      # Cache expiration
)
```

## Query Examples

### Mapping Queries
```
"Give me the mapping for field 'customer_id'"
"Show mapping for AccountNumber from source MSC"
"Find all mappings in module CRM"
"Show dimension Sales field Revenue"
"get me logics for 'event_type' where domain is RA, module is UC,
 source is MSC, vendor is Nokia and operator is DU"
```

### Layout Queries
```
"Show layout for domain RA, module UC, source MSC, vendor Nokia"
"Give me the format for domain RA, module UC, source MSC, vendor Nokia"
"Get decoder for domain RA, module UC, source MSC, vendor Nokia"
"Show layout for domain RA, module UC, source MSC, vendor Nokia, operator DU"
```

## Key Features

1. **Natural Language Processing** - Regex-based query parsing with synonym support
2. **Fuzzy Matching** - Multiple algorithms with composite confidence scoring (0-100%)
3. **Parallel Loading** - ThreadPoolExecutor with 8 workers default
4. **Smart Caching** - TTL-based with mtime validation
5. **Operator Detection** - Extracts operator from Excel filenames
6. **Markdown Output** - Formatted responses with confidence badges
7. **Layout Viewer** - View format/decoder files with operator selection
8. **Download & Copy** - Download original Excel or copy content

## Data Flow

### Mapping Query Flow
```
User Query → Gradio UI → is_layout_query()? No → parse_query() → search_mappings()
                                                                       ↓
                                                       Cache Hit? → Return cached
                                                            ↓ No
                                                       Excel Extraction → Cache → Return
                                                                       ↓
                                                       format_results() → Display
```

### Layout Query Flow
```
User Query → Gradio UI → is_layout_query()? Yes → process_layout_query()
                                                         ↓
                                              get_layout_operators()
                                                         ↓
                                              Show operator buttons
                                                         ↓
                                              User clicks operator
                                                         ↓
                                              read_excel_content() → Display in scrollable panel
```

## Commands

| Command | Description |
|---------|-------------|
| `help` | Show usage guide |
| `list` / `sources` | List available domains |
| `stats` | Show loading statistics |
| `cache stats` | Cache performance metrics |
| `clear cache` | Clear all cached files |

## Layout Feature

### How It Works
1. User asks for layout/format/decoder with domain, module, source, vendor
2. System finds matching Layout folder and extracts operators from filenames
3. Available operators are displayed as clickable buttons
4. User clicks an operator to view full content in scrollable panel
5. User can download original file or copy content

### Operator Extraction
- Filename: `Layout_Vendor_Operator.xlsx`
- Extracted operator: `Operator`
- Logic removes: "Layout" prefix, source name, vendor name, numbers

### Display Features
- Scrollable content panel (fixed height)
- Tab-separated text format for easy copying
- Download original Excel file button
- Copy content button (built into Gradio textbox)

## Development Notes

- **Port**: 8848 (default)
- **File Format**: Only `.xlsx` supported
- **Ignored**: `~$*.xlsx` (temp files), `__pycache__`, `.pyc`
- **Threading**: ThreadPoolExecutor for parallel ops
- **Serialization**: Pickle (cache), JSON (metadata)

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable (optional - has default fallback)
export MAPPING_ROOT_FOLDER="/path/to/mapping/data"

# Run the application
python src/app.py
```

Access the web interface at: `http://127.0.0.1:8848`

## Future Scope

- Hugging Face ML model integration for advanced NLP
- Database backend for scalability
- REST API layer
- Export capabilities (CSV, JSON)
- User authentication
- Query analytics
