"""
Enhanced Query Parser with Fuzzy Matching
Handles typos in keywords while keeping filter values exact.

Hierarchy: Domain → Module → Source → Vendor → Operator
"""
import re
from typing import Dict, Optional, List
from rapidfuzz import fuzz, process


class EnhancedQueryParser:
    """
    Fast, free, typo-tolerant query parser using fuzzy matching.

    Features:
    - Fuzzy matches KEYWORDS (vendor, module, domain, source, etc.) to fix typos
    - Keeps FILTER VALUES exact (Nokia, RA, DU, etc.)
    - Handles natural language variations
    - <10ms per query

    Hierarchy: Domain → Module → Source → Vendor → Operator
    """

    def __init__(self, chatbot_instance=None):
        """
        Initialize parser.

        Args:
            chatbot_instance: Reference to MappingChatBot (optional, for entity validation)
        """
        self.chatbot = chatbot_instance

        # Keyword synonyms and typo variations
        self.keywords = {
            # Action words
            'action': [
                'give', 'show', 'get', 'find', 'display', 'fetch', 'retrieve',
                'list', 'search', 'lookup', 'query', 'pull', 'extract',
                'gimme', 'giv', 'shw', 'gt', 'fnd', 'dsplay'  # Common typos
            ],

            # What user wants
            'target': [
                'logic', 'logics', 'expression', 'expressions',
                'mapping', 'mappings', 'rule', 'rules',
                'transformation', 'transformations', 'conversion', 'conversions',
                'loigc', 'logix', 'expresion', 'maping'  # Common typos
            ],

            # Field keyword
            'field': [
                'field', 'fields', 'column', 'columns', 'attribute', 'attributes',
                'fld', 'feild', 'feeld', 'colum'  # Common typos
            ],

            # Domain keyword (was 'source' in old terminology)
            'domain': [
                'domain', 'dom', 'domains',
                'doamin', 'domian', 'domein'  # Common typos
            ],

            # Module keyword
            'module': [
                'module', 'mod', 'component',
                'moduel', 'modul', 'modulee'  # Common typos
            ],

            # Source keyword (was 'source_name' in old terminology)
            'source': [
                'source', 'src', 'sources',
                'sourse', 'sorce', 'souce'  # Common typos
            ],

            # Vendor keyword
            'vendor': [
                'vendor', 'supplier', 'provider', 'vend',
                'vender', 'vendr', 'vendro'  # Common typos
            ],

            # Operator keyword
            'operator': [
                'operator', 'op', 'telco', 'telecom', 'carrier',
                'operater', 'oprator', 'opertor'  # Common typos
            ],

            # Dimension keyword
            'dimension': [
                'dimension', 'dim', 'dimensions',
                'dimesion', 'dimemsion'  # Common typos
            ],

            # Measure keyword
            'measure': [
                'measure', 'meas', 'measures', 'metric', 'metrics',
                'mesure', 'measue'  # Common typos
            ]
        }

        # Flatten for faster lookup
        self._build_keyword_map()

    def _build_keyword_map(self):
        """Build reverse lookup map: variation -> canonical_keyword"""
        self.keyword_map = {}
        for canonical, variations in self.keywords.items():
            for variation in variations:
                self.keyword_map[variation.lower()] = canonical

    def fuzzy_match_keyword(self, word: str, threshold: int = 80) -> Optional[str]:
        """
        Match a word to a known keyword using fuzzy matching.

        Args:
            word: Word to match
            threshold: Minimum similarity score (0-100)

        Returns:
            Canonical keyword or None
        """
        word_lower = word.lower().strip()

        # Quick exact lookup first
        if word_lower in self.keyword_map:
            return self.keyword_map[word_lower]

        # Fuzzy match against all variations
        all_variations = list(self.keyword_map.keys())
        result = process.extractOne(
            word_lower,
            all_variations,
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )

        if result:
            matched_variation = result[0]
            return self.keyword_map[matched_variation]

        return None

    def normalize_query(self, query: str) -> str:
        """
        Normalize query by fixing typos in keywords.

        Example:
            "vender Nokia moduel CRM" → "vendor Nokia module CRM"
            "giv me logics" → "give me logics"

        Args:
            query: User query

        Returns:
            Normalized query with fixed keywords
        """
        words = query.split()
        normalized_words = []

        for word in words:
            # Skip quoted strings (these are field names - keep exact)
            if word.startswith(("'", '"')):
                normalized_words.append(word)
                continue

            # Try to match as keyword
            matched = self.fuzzy_match_keyword(word, threshold=75)

            if matched:
                # Replace with canonical form
                if matched == 'action':
                    normalized_words.append('give')  # Standardize action verbs
                elif matched == 'target':
                    normalized_words.append('logics')  # Standardize target
                elif matched == 'field':
                    normalized_words.append('field')
                elif matched == 'domain':
                    normalized_words.append('domain')
                elif matched == 'module':
                    normalized_words.append('module')
                elif matched == 'source':
                    normalized_words.append('source')
                elif matched == 'vendor':
                    normalized_words.append('vendor')
                elif matched == 'operator':
                    normalized_words.append('operator')
                elif matched == 'dimension':
                    normalized_words.append('dimension')
                elif matched == 'measure':
                    normalized_words.append('measure')
                else:
                    normalized_words.append(word)
            else:
                # Not a keyword, keep original (could be filter value)
                normalized_words.append(word)

        return ' '.join(normalized_words)

    def parse_query(self, query: str) -> Dict[str, Optional[str]]:
        """
        Parse natural language query into structured filters.

        Process:
        1. Normalize keywords (fix typos)
        2. Extract entities using regex
        3. Return exact filter values (no fuzzy matching on values)

        Args:
            query: Natural language query

        Returns:
            Dictionary with: field, dimension, domain, module, source, vendor, operator

        Examples:
            >>> parse_query("give me logics for 'cdr_type' where vendor is Nokia")
            {'field': 'cdr_type', 'vendor': 'Nokia', ...}

            >>> parse_query("vender Nokia moduel PI")  # Typos fixed
            {'vendor': 'Nokia', 'module': 'PI', ...}
        """
        # Step 1: Normalize keywords (fix typos like vender→vendor)
        normalized_query = self.normalize_query(query)
        query_lower = normalized_query.lower()

        result = {
            'field': None,
            'dimension': None,
            'domain': None,
            'module': None,
            'source': None,
            'vendor': None,
            'operator': None
        }

        # Step 2: Extract entities using regex patterns
        # Note: We keep the extracted VALUES exact (no fuzzy matching)

        # === FIELD EXTRACTION ===
        field_patterns = [
            r"(?:for|of)\s+['\"]([^'\"]+)['\"]",  # for 'field_name'
            r"field[s]?\s+['\"]([^'\"]+)['\"]",  # field 'name'
            r"(?:logics|mapping|expression)[s]?\s+for\s+['\"]([^'\"]+)['\"]",  # logics for 'name'
            r"['\"]([^'\"]+)['\"]",  # just 'field_name'
            r"(?:for|of)\s+(\w+)(?:\s+where|\s+vendor|\s+domain|\s+source|\s+module|\s+operator|$)",  # for field_name
            r"field[s]?\s+(\w+)",  # field name
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
            r"(?:where|and)\s+(?:my\s+)?domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:in|for|from)\s+domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"domain\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in domain_patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Keep EXACT case from original query, convert to UPPERCASE
                original_match = re.search(pattern, normalized_query, re.IGNORECASE)
                if original_match:
                    result['domain'] = original_match.group(1).upper()
                else:
                    result['domain'] = match.group(1).upper()
                break

        # === MODULE EXTRACTION ===
        module_patterns = [
            r"(?:where|and)\s+(?:my\s+)?module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"(?:in|for|from)\s+module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
            r"module\s+(?:is\s+)?['\"]?([^'\".,;\s]+)['\"]?",
        ]

        for pattern in module_patterns:
            match = re.search(pattern, query_lower)
            if match:
                original_match = re.search(pattern, normalized_query, re.IGNORECASE)
                if original_match:
                    result['module'] = original_match.group(1).upper()
                else:
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
                original_match = re.search(pattern, normalized_query, re.IGNORECASE)
                if original_match:
                    result['source'] = original_match.group(1).upper()
                else:
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
                original_match = re.search(pattern, normalized_query, re.IGNORECASE)
                if original_match:
                    result['vendor'] = original_match.group(1).strip().upper()
                else:
                    result['vendor'] = match.group(1).strip().upper()
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
                original_match = re.search(pattern, normalized_query, re.IGNORECASE)
                if original_match:
                    result['operator'] = original_match.group(1).strip().upper()
                else:
                    result['operator'] = match.group(1).strip().upper()
                break

        return result

    def get_suggestions(self, query: str) -> List[str]:
        """
        Suggest corrections when query fails.

        Args:
            query: Failed query

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Check for common issues
        if "'" not in query and '"' not in query:
            suggestions.append("Try putting field name in quotes: 'field_name'")

        if not any(kw in query.lower() for kw in ['vendor', 'domain', 'source', 'module', 'operator']):
            suggestions.append("Add filters like: vendor Nokia, domain RA, module PI")

        # Check for possible typos in keywords
        words = query.lower().split()
        for word in words:
            if len(word) > 3:  # Only check substantial words
                matched = self.fuzzy_match_keyword(word, threshold=70)
                if matched and word not in self.keyword_map:
                    suggestions.append(f"Did you mean '{matched}' instead of '{word}'?")

        return suggestions[:3]  # Max 3 suggestions