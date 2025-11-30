"""
Hierarchy Query Parser for Mapping ChatBot.
Parses natural language queries into structured hierarchy query commands.

Hierarchy: Domain → Module → Source → Vendor → Operator

Supports 45+ query types across 5 categories:
1. From Domain queries
2. From Module queries
3. From Source queries
4. From Vendor queries
5. Global/System-Level queries
"""
import re
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ParsedHierarchyQuery:
    """Parsed hierarchy query structure."""
    query_type: str
    category: str  # 'domain', 'module', 'source', 'vendor', 'global'
    action: str    # 'count', 'list', 'filter', 'top', 'grouped', 'zero'
    target: str    # 'modules', 'sources', 'vendors', 'operators'
    context_type: Optional[str] = None   # 'domain', 'module', etc.
    context_value: Optional[str] = None  # 'RA', 'UC', etc.
    filter_type: Optional[str] = None    # 'min_operators', 'pattern', 'keyword'
    filter_value: Optional[Any] = None   # 5, 'DU*', 'MSC', etc.
    top_n: Optional[int] = None          # For top N queries


class HierarchyQueryParser:
    """
    Parser for natural language hierarchy queries.

    Recognizes patterns like:
    - "get number of modules under domain RA"
    - "list vendors under module UC"
    - "how many operators does vendor Nokia have"
    - "top 5 vendors by operators"
    - "vendors with more than 3 operators under domain RA"
    """

    def __init__(self):
        """Initialize parser with pattern definitions."""
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for query recognition."""

        # Action keywords
        self.count_keywords = [
            'number of', 'count of', 'how many', 'total', 'count'
        ]
        self.list_keywords = [
            'list', 'show', 'get', 'give', 'display', 'what are', 'which'
        ]

        # Target keywords (what we're looking for)
        self.target_map = {
            'modules': ['modules', 'module'],
            'sources': ['sources', 'source'],
            'vendors': ['vendors', 'vendor', 'suppliers'],
            'operators': ['operators', 'operator', 'telcos', 'carriers'],
            'domains': ['domains', 'domain']
        }

        # Context keywords (under what)
        self.context_map = {
            'domain': ['under domain', 'in domain', 'for domain', 'from domain', 'domain is', 'domain ='],
            'module': ['under module', 'in module', 'for module', 'from module', 'module is', 'module ='],
            'source': ['under source', 'in source', 'for source', 'from source', 'source is', 'source ='],
            'vendor': ['under vendor', 'in vendor', 'for vendor', 'from vendor', 'vendor is', 'vendor =']
        }

        # Filter keywords
        self.filter_patterns = {
            'min_operators': [
                r'with\s+(?:more\s+than|>|>=)\s+(\d+)\s+operators?',
                r'having\s+(?:more\s+than|>|>=)\s+(\d+)\s+operators?',
                r'(?:at\s+least|>=)\s+(\d+)\s+operators?'
            ],
            'min_sources': [
                r'with\s+(?:more\s+than|>|>=|at\s+least)\s+(\d+)\s+sources?',
                r'having\s+(?:more\s+than|>|>=|at\s+least)\s+(\d+)\s+sources?'
            ],
            'pattern': [
                r'matching\s+(?:pattern\s+)?["\']?([^"\']+)["\']?',
                r'like\s+["\']?([^"\']+)["\']?',
                r'pattern\s+["\']?([^"\']+)["\']?'
            ],
            'keyword': [
                r'containing\s+(?:keyword\s+)?["\']?([^"\']+)["\']?',
                r'with\s+keyword\s+["\']?([^"\']+)["\']?',
                r'contains\s+["\']?([^"\']+)["\']?'
            ]
        }

        # Top N patterns
        self.top_n_patterns = [
            r'top\s+(\d+)',
            r'first\s+(\d+)',
            r'(\d+)\s+(?:most|highest|largest|biggest)'
        ]

        # Grouped by patterns
        self.grouped_patterns = [
            r'grouped\s+by\s+(\w+)',
            r'group\s+by\s+(\w+)',
            r'by\s+(\w+)\s*$'
        ]

        # Zero/empty patterns
        self.zero_patterns = [
            r'with\s+(?:zero|no|0)\s+(\w+)',
            r'without\s+(?:any\s+)?(\w+)',
            r'empty\s+(\w+)',
            r'having\s+(?:zero|no|0)\s+(\w+)'
        ]

        # All unique patterns
        self.all_unique_patterns = [
            r'all\s+(?:unique\s+)?(\w+)\s*(?:names?)?',
            r'unique\s+(\w+)\s*(?:names?)?',
            r'list\s+all\s+(\w+)'
        ]

    def parse(self, query: str) -> Optional[ParsedHierarchyQuery]:
        """
        Parse a natural language query into a structured hierarchy query.
        """
        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # Try each query category in order
        result = self._try_parse_global_query(query_lower)
        if result:
            return result

        result = self._try_parse_from_domain_query(query_lower, query_stripped)
        if result:
            return result

        result = self._try_parse_from_module_query(query_lower, query_stripped)
        if result:
            return result

        result = self._try_parse_from_source_query(query_lower, query_stripped)
        if result:
            return result

        result = self._try_parse_from_vendor_query(query_lower, query_stripped)
        if result:
            return result

        return None

    def _try_parse_global_query(self, query: str) -> Optional[ParsedHierarchyQuery]:
        """Parse global/system-level queries."""

        # Total counts: "total number of domains/modules/vendors/operators"
        for target_key, target_words in self.target_map.items():
            for word in target_words:
                patterns = [
                    rf'(?:total|all)\s+(?:number\s+of\s+)?{word}',
                    rf'how\s+many\s+{word}\s+(?:are\s+there|exist|do\s+we\s+have)',
                    rf'count\s+(?:of\s+)?(?:all\s+)?{word}',
                    rf'get\s+total\s+(?:number\s+of\s+)?{word}'
                ]
                for pattern in patterns:
                    if re.search(pattern, query):
                        return ParsedHierarchyQuery(
                            query_type=f"total_{target_key}",
                            category='global',
                            action='count',
                            target=target_key
                        )

        # Top N queries: "top 5 vendors with most operators"
        for pattern in self.top_n_patterns:
            match = re.search(pattern, query)
            if match:
                n = int(match.group(1))

                if 'vendor' in query and 'operator' in query:
                    return ParsedHierarchyQuery(
                        query_type="top_vendors_by_operators",
                        category='global',
                        action='top',
                        target='vendors',
                        top_n=n
                    )
                elif 'domain' in query and 'module' in query:
                    return ParsedHierarchyQuery(
                        query_type="top_domains_by_modules",
                        category='global',
                        action='top',
                        target='domains',
                        top_n=n
                    )

        # Grouped queries: "modules grouped by domain"
        for pattern in self.grouped_patterns:
            match = re.search(pattern, query)
            if match:
                group_by = match.group(1).lower()

                if 'module' in query and group_by == 'domain':
                    return ParsedHierarchyQuery(
                        query_type="modules_grouped_by_domain",
                        category='global',
                        action='grouped',
                        target='modules',
                        filter_type='group_by',
                        filter_value='domain'
                    )
                elif 'vendor' in query and group_by == 'domain':
                    return ParsedHierarchyQuery(
                        query_type="vendors_grouped_by_domain",
                        category='global',
                        action='grouped',
                        target='vendors',
                        filter_type='group_by',
                        filter_value='domain'
                    )
                elif 'operator' in query and group_by == 'vendor':
                    return ParsedHierarchyQuery(
                        query_type="operators_grouped_by_vendor",
                        category='global',
                        action='grouped',
                        target='operators',
                        filter_type='group_by',
                        filter_value='vendor'
                    )

        # Zero/empty queries: "modules with zero sources"
        for pattern in self.zero_patterns:
            match = re.search(pattern, query)
            if match:
                zero_target = match.group(1).lower()

                if 'module' in query and 'source' in zero_target:
                    return ParsedHierarchyQuery(
                        query_type="modules_with_zero_sources",
                        category='global',
                        action='zero',
                        target='modules',
                        filter_type='zero',
                        filter_value='sources'
                    )
                elif 'source' in query and 'vendor' in zero_target:
                    return ParsedHierarchyQuery(
                        query_type="sources_with_zero_vendors",
                        category='global',
                        action='zero',
                        target='sources',
                        filter_type='zero',
                        filter_value='vendors'
                    )
                elif 'vendor' in query and 'operator' in zero_target:
                    return ParsedHierarchyQuery(
                        query_type="vendors_with_zero_operators",
                        category='global',
                        action='zero',
                        target='vendors',
                        filter_type='zero',
                        filter_value='operators'
                    )

        # All unique queries: "all unique sources"
        for pattern in self.all_unique_patterns:
            match = re.search(pattern, query)
            if match:
                target = match.group(1).lower()

                if 'source' in target:
                    return ParsedHierarchyQuery(
                        query_type="all_unique_sources",
                        category='global',
                        action='list',
                        target='sources'
                    )
                elif 'vendor' in target:
                    return ParsedHierarchyQuery(
                        query_type="all_unique_vendor_names",
                        category='global',
                        action='list',
                        target='vendors'
                    )
                elif 'operator' in target:
                    return ParsedHierarchyQuery(
                        query_type="all_unique_operator_names",
                        category='global',
                        action='list',
                        target='operators'
                    )

        return None

    def _try_parse_from_domain_query(self, query: str, original_query: str = None) -> Optional[ParsedHierarchyQuery]:
        """Parse queries starting from a domain context."""
        if original_query is None:
            original_query = query

        domain_value = self._extract_context_value(query, original_query, 'domain')
        if not domain_value:
            return None

        target, action = self._determine_target_action(query)
        if not target:
            return None

        filter_type, filter_value = self._extract_filter(query)

        if filter_type == 'min_operators' and target == 'vendors':
            return ParsedHierarchyQuery(
                query_type="vendors_with_min_operators_under_domain",
                category='domain',
                action='filter',
                target='vendors',
                context_type='domain',
                context_value=domain_value.upper(),
                filter_type='min_operators',
                filter_value=filter_value
            )
        elif filter_type == 'pattern' and target == 'operators':
            return ParsedHierarchyQuery(
                query_type="operators_matching_pattern_under_domain",
                category='domain',
                action='filter',
                target='operators',
                context_type='domain',
                context_value=domain_value.upper(),
                filter_type='pattern',
                filter_value=filter_value
            )
        elif filter_type == 'keyword' and target == 'sources':
            return ParsedHierarchyQuery(
                query_type="sources_containing_keyword_under_domain",
                category='domain',
                action='filter',
                target='sources',
                context_type='domain',
                context_value=domain_value.upper(),
                filter_type='keyword',
                filter_value=filter_value
            )
        elif filter_type == 'min_sources' and target == 'modules':
            return ParsedHierarchyQuery(
                query_type="modules_with_min_sources_under_domain",
                category='domain',
                action='filter',
                target='modules',
                context_type='domain',
                context_value=domain_value.upper(),
                filter_type='min_sources',
                filter_value=filter_value
            )
        else:
            return ParsedHierarchyQuery(
                query_type=f"{target}_{action}_under_domain",
                category='domain',
                action=action,
                target=target,
                context_type='domain',
                context_value=domain_value.upper()
            )

    def _try_parse_from_module_query(self, query: str, original_query: str = None) -> Optional[ParsedHierarchyQuery]:
        """Parse queries starting from a module context."""
        if original_query is None:
            original_query = query

        module_value = self._extract_context_value(query, original_query, 'module')
        if not module_value:
            return None

        target, action = self._determine_target_action(query)
        if not target:
            return None

        filter_type, filter_value = self._extract_filter(query)

        if filter_type == 'min_operators' and target == 'vendors':
            return ParsedHierarchyQuery(
                query_type="vendors_with_min_operators_under_module",
                category='module',
                action='filter',
                target='vendors',
                context_type='module',
                context_value=module_value.upper(),
                filter_type='min_operators',
                filter_value=filter_value
            )
        elif filter_type == 'pattern' and target == 'operators':
            return ParsedHierarchyQuery(
                query_type="operators_matching_pattern_under_module",
                category='module',
                action='filter',
                target='operators',
                context_type='module',
                context_value=module_value.upper(),
                filter_type='pattern',
                filter_value=filter_value
            )
        elif filter_type == 'keyword' and target == 'sources':
            return ParsedHierarchyQuery(
                query_type="sources_containing_keyword_under_module",
                category='module',
                action='filter',
                target='sources',
                context_type='module',
                context_value=module_value.upper(),
                filter_type='keyword',
                filter_value=filter_value
            )
        else:
            return ParsedHierarchyQuery(
                query_type=f"{target}_{action}_under_module",
                category='module',
                action=action,
                target=target,
                context_type='module',
                context_value=module_value.upper()
            )

    def _try_parse_from_source_query(self, query: str, original_query: str = None) -> Optional[ParsedHierarchyQuery]:
        """Parse queries starting from a source context."""
        if original_query is None:
            original_query = query

        source_value = self._extract_context_value(query, original_query, 'source')
        if not source_value:
            return None

        target, action = self._determine_target_action(query)
        if not target:
            return None

        filter_type, filter_value = self._extract_filter(query)

        if filter_type == 'min_operators' and target == 'vendors':
            return ParsedHierarchyQuery(
                query_type="vendors_with_min_operators_under_source",
                category='source',
                action='filter',
                target='vendors',
                context_type='source',
                context_value=source_value.upper(),
                filter_type='min_operators',
                filter_value=filter_value
            )
        elif filter_type == 'pattern' and target == 'operators':
            return ParsedHierarchyQuery(
                query_type="operators_matching_pattern_under_source",
                category='source',
                action='filter',
                target='operators',
                context_type='source',
                context_value=source_value.upper(),
                filter_type='pattern',
                filter_value=filter_value
            )
        else:
            return ParsedHierarchyQuery(
                query_type=f"{target}_{action}_under_source",
                category='source',
                action=action,
                target=target,
                context_type='source',
                context_value=source_value.upper()
            )

    def _try_parse_from_vendor_query(self, query: str, original_query: str = None) -> Optional[ParsedHierarchyQuery]:
        """Parse queries starting from a vendor context."""
        if original_query is None:
            original_query = query

        vendor_value = self._extract_context_value(query, original_query, 'vendor')
        if not vendor_value:
            return None

        target, action = self._determine_target_action(query)
        if not target or target != 'operators':
            return None

        filter_type, filter_value = self._extract_filter(query)

        if filter_type == 'pattern':
            return ParsedHierarchyQuery(
                query_type="operators_matching_pattern_under_vendor",
                category='vendor',
                action='filter',
                target='operators',
                context_type='vendor',
                context_value=vendor_value.upper(),
                filter_type='pattern',
                filter_value=filter_value
            )
        else:
            return ParsedHierarchyQuery(
                query_type=f"operators_{action}_under_vendor",
                category='vendor',
                action=action,
                target='operators',
                context_type='vendor',
                context_value=vendor_value.upper()
            )

    def _extract_context_value(self, query_lower: str, original_query: str, context_type: str) -> Optional[str]:
        """Extract the value for a context type from the query, preserving case."""

        patterns = self.context_map.get(context_type, [])

        for prefix in patterns:
            pattern = rf'{prefix}\s+["\']?([^\s"\',.]+)["\']?'
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                start, end = match.span(1)
                return original_query[start:end]

        return None

    def _determine_target_action(self, query: str) -> Tuple[Optional[str], str]:
        """Determine what target we're looking for and what action."""

        is_count = any(kw in query for kw in self.count_keywords)
        action = 'count' if is_count else 'list'

        # Find all potential targets mentioned in query with their positions
        found_targets = []
        for target_key, target_words in self.target_map.items():
            for word in target_words:
                pos = query.find(word)
                if pos != -1:
                    # Check if this word is part of a context phrase
                    is_context = False
                    for ctx_type, ctx_phrases in self.context_map.items():
                        for phrase in ctx_phrases:
                            phrase_pos = query.find(phrase)
                            if phrase_pos != -1 and phrase_pos <= pos < phrase_pos + len(phrase):
                                is_context = True
                                break
                        if is_context:
                            break

                    if not is_context:
                        found_targets.append((pos, target_key, word))

        if found_targets:
            found_targets.sort(key=lambda x: x[0])
            return found_targets[0][1], action

        return None, action

    def _extract_filter(self, query: str) -> Tuple[Optional[str], Optional[Any]]:
        """Extract filter type and value from query."""

        for pattern in self.filter_patterns['min_operators']:
            match = re.search(pattern, query)
            if match:
                return 'min_operators', int(match.group(1))

        for pattern in self.filter_patterns['min_sources']:
            match = re.search(pattern, query)
            if match:
                return 'min_sources', int(match.group(1))

        for pattern in self.filter_patterns['pattern']:
            match = re.search(pattern, query)
            if match:
                return 'pattern', match.group(1).strip()

        for pattern in self.filter_patterns['keyword']:
            match = re.search(pattern, query)
            if match:
                return 'keyword', match.group(1).strip()

        return None, None

    def is_hierarchy_query(self, query: str) -> bool:
        """Quick check if query looks like a hierarchy query."""
        query_lower = query.lower()

        indicators = [
            'number of', 'count of', 'how many', 'total',
            'list of', 'list all', 'show all',
            'under domain', 'under module', 'under vendor', 'under source',
            'grouped by', 'group by',
            'with zero', 'without any',
            'top ', 'with more than',
            'all unique', 'unique vendor', 'unique operator', 'unique source'
        ]

        return any(ind in query_lower for ind in indicators)

    def get_help_text(self) -> str:
        """Return help text for hierarchy queries."""
        return """
## Hierarchy Navigation Queries

**Hierarchy Structure:** Domain → Module → Source → Vendor → Operator

### From Domain
- "Get number of modules under domain RA"
- "List vendors under domain RA"
- "Vendors under domain RA with more than 3 operators"
- "Operators under domain RA matching pattern 'DU*'"
- "Sources under domain RA containing 'MSC'"
- "Modules under domain RA with at least 2 sources"

### From Module
- "How many vendors under module UC?"
- "List operators under module PI"
- "Sources under module UC containing 'HLR'"
- "Vendors under module UC with more than 2 operators"

### From Source
- "Number of vendors under source MSC"
- "List operators under source HLR"
- "Vendors under source MSC with more than 2 operators"

### From Vendor
- "How many operators under vendor Nokia?"
- "List operators under vendor Ericsson"
- "Operators under vendor Nokia matching pattern 'Air*'"

### Global Queries
- "Total number of domains/modules/vendors/operators"
- "Top 5 vendors with most operators"
- "Top 10 domains with most modules"
- "Modules grouped by domain"
- "Vendors grouped by domain"
- "Operators grouped by vendor"
- "Modules with zero sources"
- "Vendors with zero operators"
- "All unique sources"
- "All unique operator names"
"""