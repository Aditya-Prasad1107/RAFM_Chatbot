"""
Optimized Mapping ChatBot with parallel loading, caching, and natural language processing.
Properly tracks and displays which filename each expression comes from.
Includes Operator extraction from filenames and operator-based filtering.
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from src.extractor import extract_combined_field_mappings_from_folder
from src.cache import configure_cache, get_cache, ExcelCache


def process_single_folder(args):
    """Process a single LdRules folder (for parallel processing)"""
    folder_path, domain, module, source, vendor, cache_enabled = args

    try:
        # Get cache instance if enabled
        cache = get_cache() if cache_enabled else None

        # Always returns 3-tuple now
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
                'domain': domain,
                'module': module,
                'source': source,
                'vendor': vendor,
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
        """
        Initialize the chatbot.

        Args:
            root_folder: Root folder containing Domain/Module/Source/Vendor/LdRules structure
            use_parallel: Whether to use parallel processing for loading
            max_workers: Maximum number of parallel workers
            cache_enabled: Whether to enable caching for Excel files
            cache_dir: Directory to store cache files
            cache_size_mb: Maximum cache size in MB
            cache_ttl_hours: Time-to-live for cache entries in hours
        """
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

        # Configure cache
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

        Returns:
            List of tuples: (ld_rules_path, domain, module, source, vendor)
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
        """
        Extract operator name from filename by removing known parts.

        Logic:
        1. Remove 'LdRules' prefix (case-insensitive)
        2. Remove source (case-insensitive)
        3. Remove vendor (case-insensitive)
        4. Remove numeric values
        5. Remove common separators and clean up
        6. What remains is the operator name

        Args:
            filename: The Excel filename (without extension)
            source: The source name from folder structure
            vendor: The vendor name from folder structure

        Returns:
            Extracted operator name or "Unknown" if not found
        """
        if not filename:
            return "Unknown"

        # Start with the filename
        remaining = filename

        # Create patterns to remove (case-insensitive)
        patterns_to_remove = [
            r'(?i)^LdRules[_\-\s]*',  # LdRules at start
            r'(?i)[_\-\s]*LdRules$',  # LdRules at end
            r'(?i)[_\-\s]*LdRules[_\-\s]*',  # LdRules anywhere
        ]

        # Remove LdRules
        for pattern in patterns_to_remove:
            remaining = re.sub(pattern, '_', remaining)

        # Split by common separators
        parts = re.split(r'[_\-\s]+', remaining)

        # Filter out parts that match source, vendor, or are numeric
        filtered_parts = []
        source_lower = source.lower()
        vendor_lower = vendor.lower()

        for part in parts:
            part_lower = part.lower().strip()

            # Skip empty parts
            if not part_lower:
                continue

            # Skip if it matches source (exact or partial)
            if part_lower == source_lower or source_lower in part_lower or part_lower in source_lower:
                continue

            # Skip if it matches vendor (exact or partial)
            if part_lower == vendor_lower or vendor_lower in part_lower or part_lower in vendor_lower:
                continue

            # Skip if it's purely numeric
            if part_lower.isdigit():
                continue

            # Skip common non-operator terms
            skip_terms = {'ldrules', 'ld', 'rules', 'mapping', 'mappings', 'template', 'v1', 'v2', 'v3', 'final', 'new', 'old', 'copy'}
            if part_lower in skip_terms:
                continue

            filtered_parts.append(part)

        # Join remaining parts
        if filtered_parts:
            # Return the first meaningful part as operator (usually the operator name)
            # If multiple parts remain, join them
            operator = '_'.join(filtered_parts)
            return operator if operator else "Unknown"

        return "Unknown"

    def parse_query(self, query: str) -> Dict[str, Optional[str]]:
        """
        Enhanced natural language query parser with operator support.

        Args:
            query: Natural language query string

        Returns:
            Dictionary with extracted entities: field, domain, module, source, vendor, operator
        """
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

        # === MODULE EXTRACTION ===
        module_patterns = [
            r"(?:in|for|from)\s+module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:where|and)\s+(?:my\s+)?module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in module_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['module'] = match.group(1)
                break

        # === DOMAIN EXTRACTION ===
        domain_patterns = [
            r"(?:where|and)\s+(?:my\s+)?domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:in|for|from)\s+domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in domain_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['domain'] = match.group(1)
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
                result['source'] = match.group(1)
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
                result['vendor'] = match.group(1).strip()
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
                result['operator'] = match.group(1).strip()
                break

        return result

    def calculate_field_match_score(self, query_field: str, target_field: str) -> Tuple[float, str]:
        """
        Calculate a confidence score for field matching.

        Returns:
            Tuple of (score, match_type)
        """
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
            r'#ref!',
            r'#n/a',
            r'#value!',
            r'#name\?',
            r'#null!',
            r'#div/0!',
            r'^\s*-\s*$',
            r'^\s*$',
            r'^n/a\s*$',
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
        """
        Check if the query operator matches the extracted operator.
        Uses case-insensitive partial matching.

        Args:
            query_operator: Operator from user query
            extracted_operator: Operator extracted from filename

        Returns:
            True if operators match
        """
        if not query_operator or not extracted_operator:
            return False

        query_op_lower = query_operator.lower().strip()
        extracted_op_lower = extracted_operator.lower().strip()

        # Exact match
        if query_op_lower == extracted_op_lower:
            return True

        # Partial match (query is contained in extracted or vice versa)
        if query_op_lower in extracted_op_lower or extracted_op_lower in query_op_lower:
            return True

        return False

    def search_mappings(self, parsed_query: Dict[str, Optional[str]]) -> List[Dict]:
        """
        Search for mappings based on parsed query with improved accuracy.
        Each result includes the specific filename and operator for each expression.

        Args:
            parsed_query: Dictionary with search criteria

        Returns:
            List of matching results with confidence scores
        """
        results = []

        field = parsed_query.get('field')
        dimension = parsed_query.get('dimension')
        domain = parsed_query.get('domain')
        module = parsed_query.get('module')
        source = parsed_query.get('source')
        vendor = parsed_query.get('vendor')
        operator = parsed_query.get('operator')

        # Support multiple vendors separated by "or"
        vendors_list = []
        if vendor:
            vendors_list = [v.strip() for v in re.split(r'\s+or\s+|\s+and\s+', vendor.lower())]

        # Filter by metadata (domain, module, etc.)
        for key, mappings in self.mappings_data.items():
            dom, mod, src, vend = key

            # Apply metadata filters (case-insensitive partial matching)
            if domain and domain.lower() not in dom.lower():
                continue
            if module and module.lower() not in mod.lower():
                continue
            if source and source.lower() not in src.lower():
                continue

            # Vendor matching
            if vendor:
                vendor_matched = False
                for v in vendors_list:
                    if v in vend.lower():
                        vendor_matched = True
                        break
                if not vendor_matched:
                    continue

            # Get filename data for this vendor
            vendor_filename_info = self.filename_data.get(key, {})
            vendor_expression_filenames = self.expression_filenames_data.get(key, {})

            # Search within mappings
            for mapping_key, expressions in mappings.items():
                left_val, field_val = mapping_key

                # Filter out invalid expressions
                valid_expressions = self.filter_valid_expressions(expressions)

                if not valid_expressions:
                    continue

                # Apply field filter with confidence scoring
                field_score = 1.0
                field_match_type = "no_filter"
                if field:
                    field_score, field_match_type = self.calculate_field_match_score(field, field_val)

                    if field_score < 0.4:
                        continue

                # Apply dimension filter
                if dimension and dimension.lower() not in left_val.lower():
                    continue

                # Get expression filenames for this mapping
                expr_filenames = vendor_expression_filenames.get(mapping_key, [])
                unique_filenames = vendor_filename_info.get(mapping_key, [])

                # Calculate overall confidence score
                expression_quality_score = min(len(valid_expressions) / 5, 1.0)
                overall_score = (field_score * 0.7) + (expression_quality_score * 0.3)

                # Create individual results for each expression with its filename and operator
                for i, expression in enumerate(valid_expressions):
                    # Get the filename for this specific expression
                    if i < len(expr_filenames):
                        filename = expr_filenames[i]
                    elif unique_filenames:
                        filename = unique_filenames[0]
                    else:
                        filename = "Unknown"

                    # Extract operator from filename
                    extracted_operator = self.extract_operator_from_filename(filename, src, vend)

                    # Apply operator filter if specified
                    if operator and not self.operator_matches(operator, extracted_operator):
                        continue

                    results.append({
                        'domain': dom,
                        'module': mod,
                        'source': src,
                        'vendor': vend,
                        'dimension': left_val,
                        'field': field_val,
                        'expression': expression,
                        'filename': filename,
                        'operator': extracted_operator,
                        'confidence_score': overall_score,
                        'field_match_score': field_score,
                        'field_match_type': field_match_type,
                    })

        # Sort by confidence score (highest first), then by operator, then by filename
        results.sort(key=lambda x: (-x['confidence_score'], x['operator'], x['filename']))

        return results

    def format_results(self, results: List[Dict], max_results: int = 50) -> str:
        """
        Format search results for display with confidence scores.
        Each expression is displayed with its corresponding filename and operator.

        Args:
            results: List of search results
            max_results: Maximum number of results to display

        Returns:
            Formatted string for display
        """
        if not results:
            return "* No mappings found matching your criteria."

        total = len(results)
        display_results = results[:max_results]

        # Calculate accuracy metrics
        high_confidence_count = sum(1 for r in display_results if r['confidence_score'] >= 0.7)
        avg_confidence = sum(r['confidence_score'] for r in display_results) / len(display_results)

        response = f"**Found {total} mapping(s)**"
        if total > max_results:
            response += f" *(showing first {max_results})*"

        response += f"\n**Search Accuracy:** {avg_confidence:.1%} average confidence"
        response += f" ({high_confidence_count}/{len(display_results)} high-confidence results)\n\n"

        for idx, result in enumerate(display_results, 1):
            # Header with confidence indicator
            confidence_indicator = "HIGH" if result['confidence_score'] >= 0.8 else "MED" if result['confidence_score'] >= 0.6 else "LOW"
            response += f"### [{idx}] `{result['field']}` [{confidence_indicator}] *{result['confidence_score']:.1%} confidence*\n\n"

            # Metadata with drill-down hierarchy including operator
            response += f"- **Dimension/Measure:** `{result['dimension']}`\n"
            response += f"- **Domain:** {result['domain']} → **Module:** {result['module']} → **Source:** {result['source']} → **Vendor:** {result['vendor']} → **Operator:** {result['operator']}\n"
            response += f"- **File:** `{result['filename']}`\n"

            # Add match quality information
            response += f"- **Field Match:** {result['field_match_type'].replace('_', ' ').title()} ({result['field_match_score']:.1%})\n"

            response += "\n"

            # Expression
            response += f"**Expression:**\n\n"
            response += f"```sql\n{result['expression']}\n```\n\n"

            response += "---\n\n"

        return response

    def process_query(self, query) -> str:
        """
        Process a natural language query and return formatted results.

        Args:
            query: Natural language query string or list (from new Gradio format)

        Returns:
            Formatted response string
        """
        # Handle new Gradio 6.0.1 message format
        if isinstance(query, list) and len(query) > 0:
            last_message = query[-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                query = last_message['content']
            elif isinstance(last_message, str):
                query = last_message
            else:
                query = str(query)

        if not isinstance(query, str) or not query.strip():
            return "* Please enter a query."

        # Handle special commands
        query_lower = query.lower().strip()

        if query_lower in ['help', 'h', '?']:
            return self.get_help()

        if query_lower in ['list', 'show all', 'list all', 'sources']:
            return self.list_all_sources()

        if query_lower in ['stats', 'statistics', 'info']:
            return self.get_statistics()

        if query_lower in ['cache stats', 'cache', 'cachestats']:
            return self.get_cache_statistics()

        if query_lower in ['clear cache', 'clearcache', 'reset cache']:
            return self.clear_cache()

        # Parse and search
        parsed = self.parse_query(query)
        results = self.search_mappings(parsed)

        return self.format_results(results)

    def get_statistics(self) -> str:
        """Get statistics about loaded data."""
        total_mappings = sum(len(mappings) for mappings in self.mappings_data.values())
        total_expressions = sum(
            len(expressions)
            for mappings in self.mappings_data.values()
            for expressions in mappings.values()
        )

        response = """## Statistics

"""
        response += f"- **Load Time:** {self.load_time:.2f} seconds\n"
        response += f"- **Total Vendors:** {len(self.mappings_data)}\n"
        response += f"- **Total Field Mappings:** {total_mappings:,}\n"
        response += f"- **Total Expressions:** {total_expressions:,}\n"
        response += f"- **Parallel Processing:** {'Enabled' if self.use_parallel else 'Disabled'}\n"
        response += "\n### Top 10 by Mapping Count:\n"

        # Sort metadata by count
        sorted_meta = sorted(
            self.metadata,
            key=lambda x: x.get('count', 0),
            reverse=True
        )[:10]

        for i, meta in enumerate(sorted_meta, 1):
            response += f"\n{i}. **{meta['domain']}/{meta['module']}/{meta['vendor']}**: {meta.get('count', 0)} mappings"

        return response

    def get_cache_statistics(self) -> str:
        """Get cache performance statistics."""
        if not self.cache_enabled:
            return "## Cache Statistics\n\n* Cache is disabled."

        cache = get_cache()
        stats = cache.get_cache_stats()

        response = """## Cache Statistics

"""
        response += f"- **Cache Enabled**: Yes\n"
        response += f"- **Cache Directory**: `{stats['cache_dir']}`\n"
        response += f"- **Cache Size**: {stats['cache_size_mb']} MB\n"
        response += f"- **Cache Hits**: {stats['hits']}\n"
        response += f"- **Cache Misses**: {stats['misses']}\n"
        response += f"- **Hit Rate**: {stats['hit_rate_percent']}%\n"
        response += f"- **Total Files Processed**: {stats['total_files_processed']}\n"
        response += f"- **Errors**: {stats['errors']}\n"

        return response

    def clear_cache(self) -> str:
        """Clear all cache files."""
        if not self.cache_enabled:
            return "* Cache is disabled. Cannot clear cache."

        try:
            cache = get_cache()
            cache.clear_cache()
            return "* Cache cleared successfully! All cached Excel files have been removed."
        except Exception as e:
            return f"* Error clearing cache: {str(e)}"

    def list_all_sources(self) -> str:
        """List all available domains in the system."""
        response = "## Available Data Sources\n\n"

        # Group by domain
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
                    f"   - {meta['module']} → {meta['source']} → "
                    f"{meta['vendor']} *({meta.get('count', 0)} mappings)*\n"
                )
            response += "\n"

        return response

    def get_help(self) -> str:
        """Return help text."""
        return """## Mapping ChatBot - Help Guide

### Natural Language Queries

Ask questions naturally! The bot understands various formats:

**Example Queries:**

- "Give me the mapping for field 'customer_id'"
- "Show mapping for AccountNumber from source MSC"
- "What is the mapping for 'email' vendor Oracle"
- "Find field address in module CRM"
- "All mappings for vendor Salesforce"
- "Show dimension Sales field Revenue"
- "get me logics for 'event_type' where domain is RA, module is UC, source is MSC, vendor is Nokia and operator is DU"

### Search Filters

You can filter by any combination of:

- **Field Name** - The target field you're looking for
- **Dimension/Measure** - The left column value (Dim./Meas.)
- **Domain** - Top-level domain folder
- **Module** - Second-level module folder
- **Source** - Third-level source folder
- **Vendor** - Fourth-level vendor folder
- **Operator** - Extracted from filename (e.g., DU, Airtel, Vodafone)

### Special Commands

- `list` or `sources` - List all available domains and vendors
- `stats` - Show loading statistics and top vendors
- `cache stats` or `cache` - Show cache performance statistics
- `clear cache` - Clear all cached Excel files
- `help` - Show this help message

### Tips

* All filters are **optional** and **case-insensitive**
* Use quotes for exact matches: `'customer_id'`
* Partial matches work: "MSC" will match "MSC_PROD"
* Combine filters for precise results
* Results show dimension/measure alongside field name
* **Operator** is automatically extracted from filenames

### Understanding Results

Each result shows:
- **Dimension/Measure** - Category from column B
- **Field** - Field name from column C
- **Expression** - Mapping value from column D
- **Drill-down Hierarchy** - Domain → Module → Source → Vendor → Operator
- **File** - The Excel file where this expression was found
- **Operator** - Extracted from the filename

---

**Happy Searching!**
"""