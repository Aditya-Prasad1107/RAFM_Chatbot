"""
Optimized Mapping ChatBot with parallel loading, caching, and natural language processing.
Properly tracks and displays which filename each expression comes from.
Includes Operator extraction from filenames and operator-based filtering.

Enhanced with 45+ hierarchy navigation queries for exploring the data structure.

Hierarchy: Domain → Module → Source → Vendor → Operator
All hierarchy values are stored and displayed in UPPERCASE.
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

try:
    from src.extractor import extract_combined_field_mappings_from_folder, extract_dynamic_mapping_column_from_folder_for_pi
    from src.cache import configure_cache, get_cache, ExcelCache
    from src.enhanced_parser import EnhancedQueryParser
    from src.hierarchy_queries import HierarchyQueryEngine
    from src.hierarchy_parser import HierarchyQueryParser
except ImportError:
    from extractor import extract_combined_field_mappings_from_folder, extract_dynamic_mapping_column_from_folder_for_pi
    from cache import configure_cache, get_cache, ExcelCache
    from enhanced_parser import EnhancedQueryParser
    from hierarchy_queries import HierarchyQueryEngine
    from hierarchy_parser import HierarchyQueryParser


def process_single_folder(args):
    """Process a single LdRules folder (for parallel processing)"""
    folder_path, domain, module, source, vendor, cache_enabled = args

    try:
        cache = get_cache() if cache_enabled else None

        if module.upper() == 'PI':
            mappings, filename_info, expression_filenames = extract_dynamic_mapping_column_from_folder_for_pi(
                str(folder_path),
                cache=cache,
                use_cache=cache_enabled
            )
        else:
            mappings, filename_info, expression_filenames = extract_combined_field_mappings_from_folder(
                str(folder_path),
                cache=cache,
                use_cache=cache_enabled
            )

        return {
            'success': True,
            'key': (domain, module, source, vendor),
            'mappings': mappings,
            'filename_info': filename_info,
            'expression_filenames': expression_filenames,
            'metadata': {
                'domain': domain.upper(),
                'module': module.upper(),
                'source': source.upper(),
                'vendor': vendor.upper(),
                'path': str(folder_path),
                'count': len(mappings)
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'path': str(folder_path)
        }


class MappingChatBot:
    """
    Natural language chatbot for querying field mappings from Excel files with caching support.

    Hierarchy: Domain → Module → Source → Vendor → Operator
    All hierarchy values stored in UPPERCASE.
    """

    def __init__(
        self,
        root_folder: str,
        use_parallel: bool = True,
        max_workers: int = 4,
        cache_enabled: bool = True,
        cache_dir: str = "./cache",
        cache_size_mb: int = 500,
        cache_ttl_hours: int = 24
    ):
        self.root_folder = root_folder
        self.mappings_data: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], List[str]]] = {}
        self.filename_data: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], List[str]]] = {}
        self.expression_filenames_data: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], List[str]]] = {}
        self.metadata: List[Dict] = []
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.cache_size_mb = cache_size_mb
        self.cache_ttl_hours = cache_ttl_hours
        self.load_time = 0

        self.enhanced_parser = None
        self.hierarchy_engine = None
        self.hierarchy_parser = None

        if self.cache_enabled:
            configure_cache(
                cache_dir=self.cache_dir,
                max_cache_size_mb=self.cache_size_mb,
                enable_cache=self.cache_enabled,
                cache_ttl_hours=self.cache_ttl_hours
            )

    def scan_folder_structure(self) -> List[Tuple]:
        """
        Scan the folder structure and return all LdRules paths.
        Structure: Domain/Module/Source/Vendor/LdRules
        """
        paths_to_process = []
        root_path = Path(self.root_folder)

        if not root_path.exists():
            raise FileNotFoundError(f"Root folder not found: {self.root_folder}")

        # Navigate through Domain/Module/Source/Vendor/LdRules
        for domain_dir in root_path.iterdir():
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name

            for module_dir in domain_dir.iterdir():
                if not module_dir.is_dir():
                    continue
                module = module_dir.name

                for source_dir in module_dir.iterdir():
                    if not source_dir.is_dir():
                        continue
                    source = source_dir.name

                    for vendor_dir in source_dir.iterdir():
                        if not vendor_dir.is_dir():
                            continue
                        vendor = vendor_dir.name

                        ld_rules_path = vendor_dir / "LdRules"
                        if ld_rules_path.exists() and ld_rules_path.is_dir():
                            paths_to_process.append(
                                (ld_rules_path, domain, module, source, vendor)
                            )

        return paths_to_process

    def load_all_mappings(self):
        """Load all mappings from the folder structure."""
        start_time = time.time()
        print("\n" + "="*70)
        print("Scanning folder structure...")
        print("="*70)

        paths_to_process = self.scan_folder_structure()
        total = len(paths_to_process)

        if total == 0:
            print("* No LdRules folders found in the specified path!")
            print(f"   Searched in: {self.root_folder}")
            return

        print(f"Found {total} LdRules folder(s) to process")

        if self.use_parallel and total > 1:
            print(f"* Using parallel processing ({min(self.max_workers, total)} workers)...")
            print(f"* Cache: {'Enabled' if self.cache_enabled else 'Disabled'}")
            self._load_parallel(paths_to_process)
        else:
            print("* Loading sequentially...")
            print(f"* Cache: {'Enabled' if self.cache_enabled else 'Disabled'}")
            self._load_sequential(paths_to_process)

        self.load_time = time.time() - start_time

        print("\n" + "="*70)
        print(f"* Successfully loaded {len(self.mappings_data)} vendor(s)")
        print(f"* Total time: {self.load_time:.2f} seconds")
        print("="*70 + "\n")

        print("* Initializing enhanced query parser...")
        self.enhanced_parser = EnhancedQueryParser(self)
        print("* Enhanced parser ready!")

        print("* Initializing hierarchy query engine...")
        self.hierarchy_engine = HierarchyQueryEngine(self)
        self.hierarchy_parser = HierarchyQueryParser()
        print("* Hierarchy query engine ready! (45+ navigation queries available)")

    def _load_parallel(self, paths_to_process):
        """Load mappings in parallel using ThreadPoolExecutor."""
        workers = min(self.max_workers, len(paths_to_process))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_folder, args + (self.cache_enabled,)): args
                for args in paths_to_process
            }

            with tqdm(total=len(futures), desc="Loading", unit="folder", ncols=100) as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['success']:
                        key = result['key']
                        self.mappings_data[key] = result['mappings']
                        self.filename_data[key] = result.get('filename_info', {})
                        self.expression_filenames_data[key] = result.get('expression_filenames', {})
                        self.metadata.append(result['metadata'])
                        pbar.set_postfix({'vendor': result['metadata']['vendor'][:20]})
                    else:
                        tqdm.write(f"* Error at {result.get('path', 'Unknown')}: {result.get('error', 'Unknown')}")

                    pbar.update(1)

    def _load_sequential(self, paths_to_process):
        """Load mappings sequentially with progress bar."""
        for folder_path, domain, module, source, vendor in tqdm(
            paths_to_process,
            desc="Loading",
            unit="folder",
            ncols=100
        ):
            result = process_single_folder((folder_path, domain, module, source, vendor, self.cache_enabled))

            if result['success']:
                key = result['key']
                self.mappings_data[key] = result['mappings']
                self.filename_data[key] = result.get('filename_info', {})
                self.expression_filenames_data[key] = result.get('expression_filenames', {})
                self.metadata.append(result['metadata'])
            else:
                print(f"* Error at {result.get('path', 'Unknown')}: {result.get('error', 'Unknown')}")

    def extract_operator_from_filename(
        self,
        filename: str,
        source: str,
        vendor: str
    ) -> str:
        """Extract operator name from filename by removing known parts."""
        if not filename:
            return "UNKNOWN"

        remaining = filename

        patterns_to_remove = [
            r'(?i)^LdRules[_\-\s]*',
            r'(?i)[_\-\s]*LdRules$',
            r'(?i)[_\-\s]*LdRules[_\-\s]*',
        ]

        for pattern in patterns_to_remove:
            remaining = re.sub(pattern, '_', remaining)

        parts = re.split(r'[_\-\s]+', remaining)

        filtered_parts = []
        source_lower = source.lower()
        vendor_lower = vendor.lower()

        for part in parts:
            part_lower = part.lower().strip()

            if not part_lower:
                continue

            if part_lower == source_lower or source_lower in part_lower or part_lower in source_lower:
                continue

            if part_lower == vendor_lower or vendor_lower in part_lower or part_lower in vendor_lower:
                continue

            if part_lower.isdigit():
                continue

            skip_terms = {'ldrules', 'ld', 'rules', 'mapping', 'mappings', 'template', 'v1', 'v2', 'v3', 'final', 'new', 'old', 'copy'}
            if part_lower in skip_terms:
                continue

            filtered_parts.append(part)

        if filtered_parts:
            operator = '_'.join(filtered_parts)
            return operator.upper() if operator else "UNKNOWN"

        return "UNKNOWN"

    def parse_query(self, query: str) -> Dict[str, Optional[str]]:
        """Enhanced natural language query parser with operator support."""
        query_lower = query.lower()

        result = {
            'field': None,
            'dimension': None,
            'domain': None,
            'module': None,
            'source': None,
            'vendor': None,
            'operator': None
        }

        # === FIELD EXTRACTION ===
        field_patterns = [
            r"(?:for|of)\s+['\"]([^'\"]+)['\"]",
            r"field[s]?\s+['\"]([^'\"]+)['\"]",
            r"mapping[s]?\s+for\s+['\"]([^'\"]+)['\"]",
            r"logic[s]?\s+for\s+['\"]([^'\"]+)['\"]",
            r"['\"]([^'\"]+)['\"]",
            r"(?:for|of)\s+(\w+)",
            r"field[s]?\s+(\w+)",
        ]

        for pattern in field_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['field'] = match.group(1)
                break

        # === DIMENSION/MEASURE EXTRACTION ===
        dim_patterns = [
            r"dimension[s]?\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"measure[s]?\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"dim\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in dim_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['dimension'] = match.group(1)
                break

        # === DOMAIN EXTRACTION ===
        domain_patterns = [
            r"(?:in|for|from)\s+domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:where|and)\s+(?:my\s+)?domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in domain_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['domain'] = match.group(1).upper()
                break

        # === MODULE EXTRACTION ===
        module_patterns = [
            r"(?:in|for|from)\s+module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:where|and)\s+(?:my\s+)?module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in module_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['module'] = match.group(1).upper()
                break

        # === SOURCE EXTRACTION ===
        source_patterns = [
            r"(?:where|and)\s+(?:my\s+)?source\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:in|for|from)\s+source\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"source\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in source_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['source'] = match.group(1).upper()
                break

        # === VENDOR EXTRACTION ===
        vendor_patterns = [
            r"(?:where|and)\s+(?:my\s+)?vendor\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:in|for|from)\s+vendor\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"vendor\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in vendor_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['vendor'] = match.group(1).upper()
                break

        # === OPERATOR EXTRACTION ===
        operator_patterns = [
            r"(?:where|and)\s+(?:my\s+)?operator\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:in|for|from)\s+operator\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"operator\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in operator_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['operator'] = match.group(1).upper()
                break

        return result

    def calculate_field_match_score(self, query_field: str, target_field: str) -> Tuple[float, str]:
        """Calculate a confidence score for field matching."""
        query_norm = query_field.lower().strip()
        target_norm = target_field.lower().strip()

        if query_norm == target_norm:
            return 1.0, "exact"

        query_spaces = query_norm.replace('_', ' ')
        query_underscores = query_norm.replace(' ', '_')
        target_spaces = target_norm.replace('_', ' ')
        target_underscores = target_norm.replace(' ', '_')

        if query_norm == target_spaces or query_norm == target_underscores:
            return 0.95, "underscore_space_variation"
        if query_spaces == target_norm or query_underscores == target_norm:
            return 0.95, "underscore_space_variation"

        if target_norm.startswith(query_norm) and len(target_norm) <= len(query_norm) + 3:
            return 0.85, "prefix"
        if query_norm.startswith(target_norm) and len(query_norm) <= len(target_norm) + 3:
            return 0.85, "prefix"

        if query_norm in target_norm and len(query_norm) >= 4:
            return 0.7, "contains"
        if target_norm in query_norm and len(target_norm) >= 4:
            return 0.7, "contains"

        query_words = set(re.split(r'[_\s]+', query_norm))
        target_words = set(re.split(r'[_\s]+', target_norm))

        if query_words and target_words:
            intersection = query_words.intersection(target_words)
            union = query_words.union(target_words)
            jaccard = len(intersection) / len(union) if union else 0

            if jaccard > 0.7:
                return 0.6, "word_similarity"
            elif jaccard > 0.5:
                return 0.4, "partial_word_similarity"

        if len(query_norm) >= 3 and len(target_norm) >= 3:
            distance = self._levenshtein_distance(query_norm, target_norm)
            max_len = max(len(query_norm), len(target_norm))
            similarity = 1 - (distance / max_len)

            if similarity > 0.8:
                return 0.5, "levenshtein_close"
            elif similarity > 0.6:
                return 0.3, "levenshtein_partial"

        return 0.0, "no_match"

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def filter_valid_expressions(self, expressions: List[str]) -> List[str]:
        """Filter out invalid expressions like #REF! errors."""
        valid_expressions = []
        invalid_patterns = [
            r'#ref!', r'#n/a', r'#value!', r'#name\?',
            r'#null!', r'#div/0!', r'^\s*-\s*$', r'^\s*$', r'^n/a\s*$',
        ]

        for expr in expressions:
            expr_str = str(expr).strip()
            if not expr_str:
                continue

            is_invalid = False
            for pattern in invalid_patterns:
                if re.search(pattern, expr_str, re.IGNORECASE):
                    is_invalid = True
                    break

            if not is_invalid:
                valid_expressions.append(expr_str)

        return valid_expressions

    def operator_matches(self, query_operator: str, extracted_operator: str) -> bool:
        """Check if the query operator matches the extracted operator."""
        if not query_operator or not extracted_operator:
            return False

        query_op_upper = query_operator.upper().strip()
        extracted_op_upper = extracted_operator.upper().strip()

        if query_op_upper == extracted_op_upper:
            return True

        if query_op_upper in extracted_op_upper or extracted_op_upper in query_op_upper:
            return True

        return False

    def search_mappings(self, parsed_query: Dict[str, Optional[str]]) -> List[Dict]:
        """Search for mappings based on parsed query."""
        results = []

        field = parsed_query.get('field')
        dimension = parsed_query.get('dimension')
        domain = parsed_query.get('domain')
        module = parsed_query.get('module')
        source = parsed_query.get('source')
        vendor = parsed_query.get('vendor')
        operator = parsed_query.get('operator')

        vendors_list = []
        if vendor:
            vendors_list = [v.strip().upper() for v in re.split(r'\s+or\s+|\s+and\s+', vendor)]

        for key, mappings in self.mappings_data.items():
            dom, mod, src, vend = key

            # Apply metadata filters (case-insensitive)
            if domain and domain.upper() not in dom.upper():
                continue
            if module and module.upper() not in mod.upper():
                continue
            if source and source.upper() not in src.upper():
                continue

            if vendor:
                vendor_matched = False
                for v in vendors_list:
                    if v in vend.upper():
                        vendor_matched = True
                        break
                if not vendor_matched:
                    continue

            vendor_filename_info = self.filename_data.get(key, {})
            vendor_expression_filenames = self.expression_filenames_data.get(key, {})

            for mapping_key, expressions in mappings.items():
                left_val, field_val = mapping_key

                valid_expressions = self.filter_valid_expressions(expressions)

                if not valid_expressions:
                    continue

                field_score = 1.0
                field_match_type = "no_filter"
                if field:
                    field_score, field_match_type = self.calculate_field_match_score(field, field_val)
                    if field_score < 0.4:
                        continue

                if dimension and dimension.lower() not in left_val.lower():
                    continue

                expr_filenames = vendor_expression_filenames.get(mapping_key, [])
                unique_filenames = vendor_filename_info.get(mapping_key, [])

                expression_quality_score = min(len(valid_expressions) / 5, 1.0)
                overall_score = (field_score * 0.7) + (expression_quality_score * 0.3)

                for i, expression in enumerate(valid_expressions):
                    if i < len(expr_filenames):
                        filename = expr_filenames[i]
                    elif unique_filenames:
                        filename = unique_filenames[0]
                    else:
                        filename = "Unknown"

                    extracted_operator = self.extract_operator_from_filename(filename, src, vend)

                    if operator and not self.operator_matches(operator, extracted_operator):
                        continue

                    results.append({
                        'domain': dom.upper(),
                        'module': mod.upper(),
                        'source': src.upper(),
                        'vendor': vend.upper(),
                        'dimension': left_val,
                        'field': field_val,
                        'expression': expression,
                        'filename': filename,
                        'operator': extracted_operator.upper(),
                        'confidence_score': overall_score,
                        'field_match_score': field_score,
                        'field_match_type': field_match_type,
                    })

        results.sort(key=lambda x: (-x['confidence_score'], x['operator'], x['filename']))
        return results

    def format_results(self, results: List[Dict], max_results: int = 50) -> str:
        """Format search results for display."""
        if not results:
            return "* No mappings found matching your criteria."

        total = len(results)
        display_results = results[:max_results]

        high_confidence_count = sum(1 for r in display_results if r['confidence_score'] >= 0.7)
        avg_confidence = sum(r['confidence_score'] for r in display_results) / len(display_results)

        response = f"**Found {total} mapping(s)**"
        if total > max_results:
            response += f" *(showing first {max_results})*"

        response += f"\n**Search Accuracy:** {avg_confidence:.1%} average confidence"
        response += f" ({high_confidence_count}/{len(display_results)} high-confidence results)\n\n"

        for idx, result in enumerate(display_results, 1):
            confidence_indicator = "HIGH" if result['confidence_score'] >= 0.8 else "MED" if result['confidence_score'] >= 0.6 else "LOW"
            response += f"### [{idx}] `{result['field']}` [{confidence_indicator}] *{result['confidence_score']:.1%} confidence*\n\n"

            response += f"- **Dimension/Measure:** `{result['dimension']}`\n"
            response += f"- **Domain:** {result['domain']} → **Module:** {result['module']} → **Source:** {result['source']} → **Vendor:** {result['vendor']} → **Operator:** {result['operator']}\n"
            response += f"- **File:** `{result['filename']}`\n"
            response += f"- **Field Match:** {result['field_match_type'].replace('_', ' ').title()} ({result['field_match_score']:.1%})\n"

            response += "\n"
            response += f"**Expression:**\n\n"
            response += f"```sql\n{result['expression']}\n```\n\n"
            response += "---\n\n"

        return response

    def process_query(self, query) -> str:
        """Process a natural language query and return formatted results."""
        import ast

        # Handle various input formats from Gradio
        if isinstance(query, list) and len(query) > 0:
            last_message = query[-1]
            if isinstance(last_message, dict):
                # Format: {'content': '...', 'role': 'user'} or {'text': '...', 'type': 'text'}
                query = last_message.get('content') or last_message.get('text', str(last_message))
            elif isinstance(last_message, str):
                query = last_message
            else:
                query = str(query)

        # Handle stringified list format: "[{'text': '/help', 'type': 'text'}]"
        if isinstance(query, str) and query.startswith('[{') and query.endswith('}]'):
            try:
                parsed_list = ast.literal_eval(query)
                if isinstance(parsed_list, list) and len(parsed_list) > 0:
                    first_item = parsed_list[0]
                    if isinstance(first_item, dict):
                        query = first_item.get('text') or first_item.get('content', query)
            except (ValueError, SyntaxError):
                pass  # Keep original query if parsing fails

        if not isinstance(query, str) or not query.strip():
            return "* Please enter a query."

        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # === SPECIAL COMMANDS (start with /) ===
        if query_stripped.startswith('/'):
            return self._process_special_command(query_stripped)

        # === "GET ALL MAPPINGS" QUERY ===
        if self._is_get_all_mappings_query(query_lower):
            return self._process_get_all_mappings_query(query)

        # Try hierarchy query
        if self.hierarchy_parser and self.hierarchy_parser.is_hierarchy_query(query):
            result = self._process_hierarchy_query(query)
            if result:
                return result

        # Parse and search using enhanced parser
        if self.enhanced_parser:
            parsed = self.enhanced_parser.parse_query(query)
        else:
            parsed = self.parse_query(query)

        results = self.search_mappings(parsed)
        return self.format_results(results)

    def _process_special_command(self, command: str) -> str:
        """Process special commands starting with /"""
        cmd = command.lower().strip()

        # /help
        if cmd in ['/help', '/h', '/?']:
            return self.get_help()

        # /list or /domains
        if cmd in ['/list', '/domains']:
            return self.list_all_domains()

        # /stats
        if cmd in ['/stats', '/statistics', '/info']:
            return self.get_statistics()

        # /hierarchy
        if cmd in ['/hierarchy', '/nav', '/navigation']:
            return self.get_hierarchy_help()

        # /vendors
        if cmd == '/vendors':
            return self.list_all_vendors()

        # /operators
        if cmd == '/operators':
            return self.list_all_operators()

        # /modules
        if cmd == '/modules':
            return self.list_all_modules()

        # /sources
        if cmd == '/sources':
            return self.list_all_sources()

        # /examples
        if cmd in ['/examples', '/example']:
            return self.get_examples()

        # Unknown command
        return f"❌ Unknown command: `{command}`\n\nType `/help` to see available commands."

    def _is_get_all_mappings_query(self, query_lower: str) -> bool:
        """Check if query is a 'get all mappings' type query."""
        patterns = [
            r'get\s+all\s+mappings',
            r'show\s+all\s+mappings',
            r'list\s+all\s+mappings',
            r'all\s+mappings\s+(?:for|where|from)',
            r'give\s+(?:me\s+)?all\s+mappings',
        ]
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _process_get_all_mappings_query(self, query: str) -> str:
        """Process 'get all mappings' query with validation."""
        # Parse the query to extract filters
        if self.enhanced_parser:
            parsed = self.enhanced_parser.parse_query(query)
        else:
            parsed = self.parse_query(query)

        domain = parsed.get('domain')
        module = parsed.get('module')
        source = parsed.get('source')
        vendor = parsed.get('vendor')
        operator = parsed.get('operator')

        # Validation: Domain + Module + Source + Vendor are ALL required
        missing = []
        if not domain:
            missing.append('Domain')
        if not module:
            missing.append('Module')
        if not source:
            missing.append('Source')
        if not vendor:
            missing.append('Vendor')

        if missing:
            return (
                f"❌ **Error:** Missing required filters: **{', '.join(missing)}**\n\n"
                f"To get all mappings, you must specify **Domain + Module + Source + Vendor** (Operator is optional).\n\n"
                f"**Example queries:**\n"
                f"- `Get all mappings where domain is RA, module is UC, source is MSC, vendor is Nokia`\n"
                f"- `Get all mappings where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU`"
            )

        # Search for all mappings (no field filter)
        results = self._search_all_mappings(domain, module, source, vendor, operator)

        if not results:
            filter_str = f"Domain={domain}, Module={module}, Source={source}, Vendor={vendor}"
            if operator:
                filter_str += f", Operator={operator}"
            return f"❌ No mappings found for: {filter_str}"

        # Format as table
        return self._format_all_mappings_table(results, domain, module, source, vendor, operator)

    def _search_all_mappings(
        self,
        domain: str,
        module: str,
        source: str,
        vendor: str,
        operator: Optional[str] = None
    ) -> List[Dict]:
        """Search for all mappings matching the given filters (no field filter)."""
        results = []

        for key, mappings in self.mappings_data.items():
            dom, mod, src, vend = key

            # Apply filters (case-insensitive)
            if domain.upper() not in dom.upper():
                continue
            if module.upper() not in mod.upper():
                continue
            if source.upper() not in src.upper():
                continue
            if vendor.upper() not in vend.upper():
                continue

            vendor_filename_info = self.filename_data.get(key, {})
            vendor_expression_filenames = self.expression_filenames_data.get(key, {})

            for mapping_key, expressions in mappings.items():
                left_val, field_val = mapping_key

                valid_expressions = self.filter_valid_expressions(expressions)

                if not valid_expressions:
                    continue

                expr_filenames = vendor_expression_filenames.get(mapping_key, [])
                unique_filenames = vendor_filename_info.get(mapping_key, [])

                for i, expression in enumerate(valid_expressions):
                    if i < len(expr_filenames):
                        filename = expr_filenames[i]
                    elif unique_filenames:
                        filename = unique_filenames[0]
                    else:
                        filename = "Unknown"

                    extracted_operator = self.extract_operator_from_filename(filename, src, vend)

                    # Filter by operator if specified
                    if operator and not self.operator_matches(operator, extracted_operator):
                        continue

                    results.append({
                        'dimension': left_val,
                        'field': field_val,
                        'expression': expression,
                        'operator': extracted_operator.upper(),
                        'filename': filename,
                    })

        # Sort by dimension, then field
        results.sort(key=lambda x: (x['dimension'], x['field']))
        return results

    def _format_all_mappings_table(
        self,
        results: List[Dict],
        domain: str,
        module: str,
        source: str,
        vendor: str,
        operator: Optional[str] = None
    ) -> str:
        """Format all mappings as a table."""
        filter_str = f"**Domain:** {domain.upper()} → **Module:** {module.upper()} → **Source:** {source.upper()} → **Vendor:** {vendor.upper()}"
        if operator:
            filter_str += f" → **Operator:** {operator.upper()}"

        response = f"## All Mappings\n\n{filter_str}\n\n"
        response += f"**Total:** {len(results)} mapping(s)\n\n"

        # Table header
        response += "| Dimension/Measure/Detailed | Field | Expression |\n"
        response += "|----------------------------|-------|------------|\n"

        # Table rows
        for result in results:
            # Escape pipe characters in expression and truncate if too long
            expr = result['expression'].replace('|', '\\|').replace('\n', ' ')
            if len(expr) > 100:
                expr = expr[:97] + "..."

            response += f"| {result['dimension']} | {result['field']} | `{expr}` |\n"

        return response

    def _process_hierarchy_query(self, query: str) -> Optional[str]:
        """Process a hierarchy navigation query."""
        if not self.hierarchy_parser or not self.hierarchy_engine:
            return None

        parsed = self.hierarchy_parser.parse(query)
        if not parsed:
            return None

        result = self._execute_hierarchy_query(parsed)

        if result:
            return self.hierarchy_engine.format_result(result)

        return None

    def _execute_hierarchy_query(self, parsed):
        """Execute a parsed hierarchy query."""
        qt = parsed.query_type
        engine = self.hierarchy_engine

        # Global queries
        if qt == "total_domains":
            return engine.get_total_domains()
        elif qt == "total_modules":
            return engine.get_total_modules()
        elif qt == "total_sources":
            return engine.get_total_sources()
        elif qt == "total_vendors":
            return engine.get_total_vendors()
        elif qt == "total_operators":
            return engine.get_total_operators()
        elif qt == "modules_grouped_by_domain":
            return engine.get_modules_grouped_by_domain()
        elif qt == "vendors_grouped_by_domain":
            return engine.get_vendors_grouped_by_domain()
        elif qt == "operators_grouped_by_vendor":
            return engine.get_operators_grouped_by_vendor()
        elif qt == "top_vendors_by_operators":
            return engine.get_top_vendors_by_operators(parsed.top_n or 10)
        elif qt == "top_domains_by_modules":
            return engine.get_top_domains_by_modules(parsed.top_n or 10)
        elif qt == "modules_with_zero_sources":
            return engine.get_modules_with_zero_sources()
        elif qt == "sources_with_zero_vendors":
            return engine.get_sources_with_zero_vendors()
        elif qt == "vendors_with_zero_operators":
            return engine.get_vendors_with_zero_operators()
        elif qt == "all_unique_sources":
            return engine.get_all_unique_sources()
        elif qt == "all_unique_vendor_names":
            return engine.get_all_unique_vendor_names()
        elif qt == "all_unique_operator_names":
            return engine.get_all_unique_operator_names()

        # From Domain queries
        elif qt == "modules_count_under_domain" or qt == "modules_list_under_domain":
            return engine.get_modules_count_under_domain(parsed.context_value)
        elif qt == "sources_count_under_domain" or qt == "sources_list_under_domain":
            return engine.get_sources_count_under_domain(parsed.context_value)
        elif qt == "vendors_count_under_domain" or qt == "vendors_list_under_domain":
            return engine.get_vendors_count_under_domain(parsed.context_value)
        elif qt == "operators_count_under_domain" or qt == "operators_list_under_domain":
            return engine.get_operators_count_under_domain(parsed.context_value)
        elif qt == "vendors_with_min_operators_under_domain":
            return engine.get_vendors_with_min_operators_under_domain(
                parsed.context_value, parsed.filter_value
            )
        elif qt == "operators_matching_pattern_under_domain":
            return engine.get_operators_matching_pattern_under_domain(
                parsed.context_value, parsed.filter_value
            )
        elif qt == "sources_containing_keyword_under_domain":
            return engine.get_sources_containing_keyword_under_domain(
                parsed.context_value, parsed.filter_value
            )
        elif qt == "modules_with_min_sources_under_domain":
            return engine.get_modules_with_min_sources_under_domain(
                parsed.context_value, parsed.filter_value
            )

        # From Module queries
        elif qt == "sources_count_under_module" or qt == "sources_list_under_module":
            return engine.get_sources_count_under_module(parsed.context_value)
        elif qt == "vendors_count_under_module" or qt == "vendors_list_under_module":
            return engine.get_vendors_count_under_module(parsed.context_value)
        elif qt == "operators_count_under_module" or qt == "operators_list_under_module":
            return engine.get_operators_count_under_module(parsed.context_value)
        elif qt == "sources_containing_keyword_under_module":
            return engine.get_sources_containing_keyword_under_module(
                parsed.context_value, parsed.filter_value
            )
        elif qt == "vendors_with_min_operators_under_module":
            return engine.get_vendors_with_min_operators_under_module(
                parsed.context_value, parsed.filter_value
            )
        elif qt == "operators_matching_pattern_under_module":
            return engine.get_operators_matching_pattern_under_module(
                parsed.context_value, parsed.filter_value
            )

        # From Source queries
        elif qt == "vendors_count_under_source" or qt == "vendors_list_under_source":
            return engine.get_vendors_count_under_source(parsed.context_value)
        elif qt == "operators_count_under_source" or qt == "operators_list_under_source":
            return engine.get_operators_count_under_source(parsed.context_value)
        elif qt == "vendors_with_min_operators_under_source":
            return engine.get_vendors_with_min_operators_under_source(
                parsed.context_value, parsed.filter_value
            )
        elif qt == "operators_matching_pattern_under_source":
            return engine.get_operators_matching_pattern_under_source(
                parsed.context_value, parsed.filter_value
            )

        # From Vendor queries
        elif qt == "operators_count_under_vendor" or qt == "operators_list_under_vendor":
            return engine.get_operators_count_under_vendor(parsed.context_value)
        elif qt == "operators_matching_pattern_under_vendor":
            return engine.get_operators_matching_pattern_under_vendor(
                parsed.context_value, parsed.filter_value
            )

        return None

    def get_statistics(self) -> str:
        """Get statistics about loaded data."""
        total_mappings = sum(len(mappings) for mappings in self.mappings_data.values())
        total_expressions = sum(
            len(expressions)
            for mappings in self.mappings_data.values()
            for expressions in mappings.values()
        )

        response = "## 📊 Statistics\n\n"
        response += f"- **Load Time:** {self.load_time:.2f} seconds\n"
        response += f"- **Total Vendors:** {len(self.mappings_data)}\n"
        response += f"- **Total Field Mappings:** {total_mappings:,}\n"
        response += f"- **Total Expressions:** {total_expressions:,}\n"

        if self.hierarchy_engine:
            response += f"\n### Hierarchy Summary\n"
            response += f"- **Domains:** {len(self.hierarchy_engine.domains)}\n"
            response += f"- **Modules:** {len(self.hierarchy_engine.modules)}\n"
            response += f"- **Sources:** {len(self.hierarchy_engine.sources)}\n"
            response += f"- **Vendors:** {len(self.hierarchy_engine.vendors)}\n"
            response += f"- **Operators:** {len(self.hierarchy_engine.operators)}\n"

        return response

    def list_all_domains(self) -> str:
        """List all available domains in the system."""
        response = "## 🏢 Available Domains\n\n"

        by_domain = {}
        for meta in self.metadata:
            domain = meta['domain']
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(meta)

        for domain, items in sorted(by_domain.items()):
            response += f"### **{domain}**\n\n"
            for meta in items:
                response += (
                    f"- {meta['module']} → {meta['source']} → "
                    f"{meta['vendor']} *({meta.get('count', 0)} mappings)*\n"
                )
            response += "\n"

        return response

    def list_all_vendors(self) -> str:
        """List all unique vendors."""
        if not self.hierarchy_engine:
            return "❌ Hierarchy engine not initialized."

        vendors = sorted(self.hierarchy_engine.vendors)
        response = f"## 🏭 All Vendors ({len(vendors)})\n\n"
        for vendor in vendors:
            response += f"- {vendor}\n"
        return response

    def list_all_operators(self) -> str:
        """List all unique operators."""
        if not self.hierarchy_engine:
            return "❌ Hierarchy engine not initialized."

        operators = sorted(self.hierarchy_engine.operators)
        response = f"## 📡 All Operators ({len(operators)})\n\n"
        for operator in operators:
            response += f"- {operator}\n"
        return response

    def list_all_modules(self) -> str:
        """List all unique modules."""
        if not self.hierarchy_engine:
            return "❌ Hierarchy engine not initialized."

        modules = sorted(self.hierarchy_engine.modules)
        response = f"## 📦 All Modules ({len(modules)})\n\n"
        for module in modules:
            response += f"- {module}\n"
        return response

    def list_all_sources(self) -> str:
        """List all unique sources."""
        if not self.hierarchy_engine:
            return "❌ Hierarchy engine not initialized."

        sources = sorted(self.hierarchy_engine.sources)
        response = f"## 📂 All Sources ({len(sources)})\n\n"
        for source in sources:
            response += f"- {source}\n"
        return response

    def get_hierarchy_help(self) -> str:
        """Return help text for hierarchy navigation queries."""
        if self.hierarchy_parser:
            return self.hierarchy_parser.get_help_text()
        return "Hierarchy query engine not initialized."

    def get_examples(self) -> str:
        """Return example queries."""
        return """## 📝 Example Queries

### Field Mapping Queries
```
Give me the mapping for field 'customer_id'
Show mapping for AccountNumber from domain RA
What is the mapping for 'email' vendor Oracle
Find all mappings in module CRM
get me logics for 'event_type' where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU
```

### Get All Mappings (Table Format)
```
Get all mappings where domain is RA, module is UC, source is MSC, vendor is Nokia
Get all mappings where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU
Show all mappings for domain RA module UC source MSC vendor Ericsson
```
*Note: Domain + Module + Source + Vendor are required. Operator is optional.*

### Hierarchy Navigation Queries
```
How many modules under domain RA?
List vendors under domain RA
Total number of operators
Top 5 vendors with most operators
Modules grouped by domain
Vendors with zero operators
```

### Special Commands
```
/help       - Show help guide
/list       - List all domains
/stats      - Show statistics
/hierarchy  - Hierarchy navigation help
/vendors    - List all vendors
/operators  - List all operators
/modules    - List all modules
/sources    - List all sources
/examples   - Show this examples list
```
"""

    def get_help(self) -> str:
        """Return help text."""
        return """## 📚 Mapping ChatBot - Help Guide

### Hierarchy Structure
**Domain → Module → Source → Vendor → Operator**

All hierarchy values are displayed in UPPERCASE.

### Natural Language Queries

Ask questions naturally! The bot understands various formats:

**Example Queries:**
- "Give me the mapping for field 'customer_id'"
- "Show mapping for AccountNumber from domain RA"
- "What is the mapping for 'email' vendor Oracle"
- "Find field address in module CRM"
- "get me logics for 'event_type' where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU"

### Get All Mappings (Table Format)

Get all mappings for a specific vendor combination without specifying a field:
- "Get all mappings where domain is RA, module is UC, source is MSC, vendor is Nokia"
- "Get all mappings where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU"

**Required:** Domain + Module + Source + Vendor (all 4 mandatory)
**Optional:** Operator

### Search Filters

You can filter by any combination of:
- **Field Name** - The target field you're looking for
- **Dimension/Measure** - The left column value
- **Domain** - Top-level domain folder (e.g., RA, FM)
- **Module** - Second-level module folder (e.g., UC, PI)
- **Source** - Third-level source folder (e.g., MSC, HLR)
- **Vendor** - Fourth-level vendor folder (e.g., Nokia, Ericsson)
- **Operator** - Extracted from filename (e.g., DU, AIRTEL)

### Hierarchy Navigation Queries

Explore the data structure with queries like:
- "How many modules under domain RA?"
- "List vendors under module UC"
- "Top 5 vendors by operator count"
- "Modules grouped by domain"
- "Vendors with zero operators"

### Special Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help guide |
| `/list` | List all domains with hierarchy |
| `/stats` | Show statistics |
| `/hierarchy` | Hierarchy navigation help |
| `/vendors` | List all vendors |
| `/operators` | List all operators |
| `/modules` | List all modules |
| `/sources` | List all sources |
| `/examples` | Show example queries |

---
**Tip:** Type `/examples` to see sample queries!
"""