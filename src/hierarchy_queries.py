"""
Hierarchy Query Engine for Mapping ChatBot.
Provides 45+ query capabilities for navigating the Source/Module/SourceName/Vendor/Operator hierarchy.

Query Categories:
1. From Source (12 queries)
2. From Module (9 queries)
3. From Source Name (6 queries)
4. From Vendor (3 queries)
5. Global/System-Level (15 queries)
"""
import re
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class HierarchyQueryResult:
    """Result container for hierarchy queries."""
    success: bool
    query_type: str
    title: str
    data: Any
    count: Optional[int] = None
    message: Optional[str] = None


class HierarchyQueryEngine:
    """
    Engine for executing hierarchy navigation queries.

    Works with the MappingChatBot's data structures to provide
    aggregation, counting, listing, and filtering capabilities.
    """

    def __init__(self, chatbot_instance):
        """
        Initialize with reference to the chatbot.

        Args:
            chatbot_instance: MappingChatBot instance with loaded data
        """
        self.chatbot = chatbot_instance
        self._build_indexes()

    def _build_indexes(self):
        """Build inverted indexes for fast hierarchy queries."""
        # Primary indexes
        self.sources: Set[str] = set()
        self.modules: Set[str] = set()
        self.source_names: Set[str] = set()
        self.vendors: Set[str] = set()
        self.operators: Set[str] = set()

        # Relationship indexes
        self.source_to_modules: Dict[str, Set[str]] = defaultdict(set)
        self.source_to_source_names: Dict[str, Set[str]] = defaultdict(set)
        self.source_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.source_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.module_to_source_names: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.source_name_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.source_name_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.vendor_to_operators: Dict[str, Set[str]] = defaultdict(set)

        # Reverse indexes for grouping
        self.operator_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.vendor_to_source_names: Dict[str, Set[str]] = defaultdict(set)
        self.source_name_to_modules: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_sources: Dict[str, Set[str]] = defaultdict(set)

        # Full path tracking for operator extraction
        self.full_paths: List[Tuple[str, str, str, str]] = []

        # Build indexes from metadata
        for meta in self.chatbot.metadata:
            source = meta['source']
            module = meta['module']
            source_name = meta['source_name']
            vendor = meta['vendor']

            self.sources.add(source)
            self.modules.add(module)
            self.source_names.add(source_name)
            self.vendors.add(vendor)

            self.source_to_modules[source].add(module)
            self.source_to_source_names[source].add(source_name)
            self.source_to_vendors[source].add(vendor)

            self.module_to_source_names[module].add(source_name)
            self.module_to_vendors[module].add(vendor)

            self.source_name_to_vendors[source_name].add(vendor)

            # Reverse indexes
            self.module_to_sources[module].add(source)
            self.source_name_to_modules[source_name].add(module)
            self.vendor_to_source_names[vendor].add(source_name)

            self.full_paths.append((source, module, source_name, vendor))

        # Extract operators from filenames
        self._build_operator_indexes()

    def _build_operator_indexes(self):
        """Build operator indexes by extracting from filenames."""
        for key, expr_filenames in self.chatbot.expression_filenames_data.items():
            source, module, source_name, vendor = key

            # Get unique filenames for this vendor
            unique_filenames = set()
            for filenames in expr_filenames.values():
                unique_filenames.update(filenames)

            # Also check filename_data
            if key in self.chatbot.filename_data:
                for filenames in self.chatbot.filename_data[key].values():
                    unique_filenames.update(filenames)

            # Extract operators from each filename
            for filename in unique_filenames:
                operator = self.chatbot.extract_operator_from_filename(
                    filename, source_name, vendor
                )
                if operator and operator != "Unknown":
                    self.operators.add(operator)
                    self.source_to_operators[source].add(operator)
                    self.module_to_operators[module].add(operator)
                    self.source_name_to_operators[source_name].add(operator)
                    self.vendor_to_operators[vendor].add(operator)
                    self.operator_to_vendors[operator].add(vendor)

    def refresh_indexes(self):
        """Refresh all indexes (call after data reload)."""
        self._build_indexes()

    # =========================================================================
    # FROM SOURCE QUERIES (12)
    # =========================================================================

    def get_modules_count_under_source(self, source: str) -> HierarchyQueryResult:
        """Get number of modules under a source."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="modules_count_under_source",
                title=f"Modules under Source '{source}'",
                data=None,
                message=f"Source '{source}' not found. Available sources: {', '.join(sorted(self.sources))}"
            )

        modules = self.source_to_modules.get(source_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="modules_count_under_source",
            title=f"Number of Modules under Source '{source_key}'",
            data=sorted(modules),
            count=len(modules)
        )

    def get_source_names_count_under_source(self, source: str) -> HierarchyQueryResult:
        """Get number of source names under a source."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="source_names_count_under_source",
                title=f"Source Names under Source '{source}'",
                data=None,
                message=f"Source '{source}' not found. Available sources: {', '.join(sorted(self.sources))}"
            )

        source_names = self.source_to_source_names.get(source_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="source_names_count_under_source",
            title=f"Number of Source Names under Source '{source_key}'",
            data=sorted(source_names),
            count=len(source_names)
        )

    def get_vendors_count_under_source(self, source: str) -> HierarchyQueryResult:
        """Get number of vendors under a source."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="vendors_count_under_source",
                title=f"Vendors under Source '{source}'",
                data=None,
                message=f"Source '{source}' not found. Available sources: {', '.join(sorted(self.sources))}"
            )

        vendors = self.source_to_vendors.get(source_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="vendors_count_under_source",
            title=f"Number of Vendors under Source '{source_key}'",
            data=sorted(vendors),
            count=len(vendors)
        )

    def get_operators_count_under_source(self, source: str) -> HierarchyQueryResult:
        """Get number of operators under a source."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_count_under_source",
                title=f"Operators under Source '{source}'",
                data=None,
                message=f"Source '{source}' not found. Available sources: {', '.join(sorted(self.sources))}"
            )

        operators = self.source_to_operators.get(source_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="operators_count_under_source",
            title=f"Number of Operators under Source '{source_key}'",
            data=sorted(operators),
            count=len(operators)
        )

    def get_modules_list_under_source(self, source: str) -> HierarchyQueryResult:
        """Get list of modules under a source."""
        return self.get_modules_count_under_source(source)

    def get_source_names_list_under_source(self, source: str) -> HierarchyQueryResult:
        """Get list of source names under a source."""
        return self.get_source_names_count_under_source(source)

    def get_vendors_list_under_source(self, source: str) -> HierarchyQueryResult:
        """Get list of vendors under a source."""
        return self.get_vendors_count_under_source(source)

    def get_operators_list_under_source(self, source: str) -> HierarchyQueryResult:
        """Get list of operators under a source."""
        return self.get_operators_count_under_source(source)

    def get_vendors_with_min_operators_under_source(
            self, source: str, min_operators: int
    ) -> HierarchyQueryResult:
        """Get vendors under a source with more than X operators."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="vendors_with_min_operators_under_source",
                title=f"Vendors under Source '{source}' with >{min_operators} operators",
                data=None,
                message=f"Source '{source}' not found."
            )

        # Get vendors under this source
        vendors = self.source_to_vendors.get(source_key, set())

        # Filter by operator count
        matching_vendors = {}
        for vendor in vendors:
            # Get operators for this vendor that are also under this source
            vendor_operators = self.vendor_to_operators.get(vendor, set())
            source_operators = self.source_to_operators.get(source_key, set())
            common_operators = vendor_operators & source_operators

            if len(common_operators) > min_operators:
                matching_vendors[vendor] = sorted(common_operators)

        return HierarchyQueryResult(
            success=True,
            query_type="vendors_with_min_operators_under_source",
            title=f"Vendors under Source '{source_key}' with more than {min_operators} operators",
            data=matching_vendors,
            count=len(matching_vendors)
        )

    def get_operators_matching_pattern_under_source(
            self, source: str, pattern: str
    ) -> HierarchyQueryResult:
        """Get operators under a source matching a pattern."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_matching_pattern_under_source",
                title=f"Operators under Source '{source}' matching '{pattern}'",
                data=None,
                message=f"Source '{source}' not found."
            )

        operators = self.source_to_operators.get(source_key, set())
        matching = self._filter_by_pattern(operators, pattern)

        return HierarchyQueryResult(
            success=True,
            query_type="operators_matching_pattern_under_source",
            title=f"Operators under Source '{source_key}' matching '{pattern}'",
            data=sorted(matching),
            count=len(matching)
        )

    def get_source_names_containing_keyword_under_source(
            self, source: str, keyword: str
    ) -> HierarchyQueryResult:
        """Get source names under a source containing a keyword."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="source_names_containing_keyword_under_source",
                title=f"Source Names under Source '{source}' containing '{keyword}'",
                data=None,
                message=f"Source '{source}' not found."
            )

        source_names = self.source_to_source_names.get(source_key, set())
        matching = [sn for sn in source_names if keyword.lower() in sn.lower()]

        return HierarchyQueryResult(
            success=True,
            query_type="source_names_containing_keyword_under_source",
            title=f"Source Names under Source '{source_key}' containing '{keyword}'",
            data=sorted(matching),
            count=len(matching)
        )

    def get_modules_with_min_source_names_under_source(
            self, source: str, min_source_names: int
    ) -> HierarchyQueryResult:
        """Get modules under a source with at least X source names."""
        source_key = self._find_matching_key(source, self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False,
                query_type="modules_with_min_source_names_under_source",
                title=f"Modules under Source '{source}' with >={min_source_names} source names",
                data=None,
                message=f"Source '{source}' not found."
            )

        modules = self.source_to_modules.get(source_key, set())

        # Count source names per module under this source
        matching_modules = {}
        for module in modules:
            # Get source names that belong to both this module AND this source
            module_source_names = self.module_to_source_names.get(module, set())
            source_source_names = self.source_to_source_names.get(source_key, set())
            common = module_source_names & source_source_names

            if len(common) >= min_source_names:
                matching_modules[module] = sorted(common)

        return HierarchyQueryResult(
            success=True,
            query_type="modules_with_min_source_names_under_source",
            title=f"Modules under Source '{source_key}' with at least {min_source_names} source names",
            data=matching_modules,
            count=len(matching_modules)
        )

    # =========================================================================
    # FROM MODULE QUERIES (9)
    # =========================================================================

    def get_source_names_count_under_module(self, module: str) -> HierarchyQueryResult:
        """Get number of source names under a module."""
        module_key = self._find_matching_key(module, self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False,
                query_type="source_names_count_under_module",
                title=f"Source Names under Module '{module}'",
                data=None,
                message=f"Module '{module}' not found. Available modules: {', '.join(sorted(self.modules))}"
            )

        source_names = self.module_to_source_names.get(module_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="source_names_count_under_module",
            title=f"Number of Source Names under Module '{module_key}'",
            data=sorted(source_names),
            count=len(source_names)
        )

    def get_vendors_count_under_module(self, module: str) -> HierarchyQueryResult:
        """Get number of vendors under a module."""
        module_key = self._find_matching_key(module, self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False,
                query_type="vendors_count_under_module",
                title=f"Vendors under Module '{module}'",
                data=None,
                message=f"Module '{module}' not found. Available modules: {', '.join(sorted(self.modules))}"
            )

        vendors = self.module_to_vendors.get(module_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="vendors_count_under_module",
            title=f"Number of Vendors under Module '{module_key}'",
            data=sorted(vendors),
            count=len(vendors)
        )

    def get_operators_count_under_module(self, module: str) -> HierarchyQueryResult:
        """Get number of operators under a module."""
        module_key = self._find_matching_key(module, self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_count_under_module",
                title=f"Operators under Module '{module}'",
                data=None,
                message=f"Module '{module}' not found. Available modules: {', '.join(sorted(self.modules))}"
            )

        operators = self.module_to_operators.get(module_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="operators_count_under_module",
            title=f"Number of Operators under Module '{module_key}'",
            data=sorted(operators),
            count=len(operators)
        )

    def get_source_names_list_under_module(self, module: str) -> HierarchyQueryResult:
        """Get list of source names under a module."""
        return self.get_source_names_count_under_module(module)

    def get_vendors_list_under_module(self, module: str) -> HierarchyQueryResult:
        """Get list of vendors under a module."""
        return self.get_vendors_count_under_module(module)

    def get_operators_list_under_module(self, module: str) -> HierarchyQueryResult:
        """Get list of operators under a module."""
        return self.get_operators_count_under_module(module)

    def get_source_names_containing_keyword_under_module(
            self, module: str, keyword: str
    ) -> HierarchyQueryResult:
        """Get source names under a module containing a keyword."""
        module_key = self._find_matching_key(module, self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False,
                query_type="source_names_containing_keyword_under_module",
                title=f"Source Names under Module '{module}' containing '{keyword}'",
                data=None,
                message=f"Module '{module}' not found."
            )

        source_names = self.module_to_source_names.get(module_key, set())
        matching = [sn for sn in source_names if keyword.lower() in sn.lower()]

        return HierarchyQueryResult(
            success=True,
            query_type="source_names_containing_keyword_under_module",
            title=f"Source Names under Module '{module_key}' containing '{keyword}'",
            data=sorted(matching),
            count=len(matching)
        )

    def get_vendors_with_min_operators_under_module(
            self, module: str, min_operators: int
    ) -> HierarchyQueryResult:
        """Get vendors under a module with more than X operators."""
        module_key = self._find_matching_key(module, self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False,
                query_type="vendors_with_min_operators_under_module",
                title=f"Vendors under Module '{module}' with >{min_operators} operators",
                data=None,
                message=f"Module '{module}' not found."
            )

        vendors = self.module_to_vendors.get(module_key, set())

        matching_vendors = {}
        for vendor in vendors:
            vendor_operators = self.vendor_to_operators.get(vendor, set())
            module_operators = self.module_to_operators.get(module_key, set())
            common_operators = vendor_operators & module_operators

            if len(common_operators) > min_operators:
                matching_vendors[vendor] = sorted(common_operators)

        return HierarchyQueryResult(
            success=True,
            query_type="vendors_with_min_operators_under_module",
            title=f"Vendors under Module '{module_key}' with more than {min_operators} operators",
            data=matching_vendors,
            count=len(matching_vendors)
        )

    def get_operators_matching_pattern_under_module(
            self, module: str, pattern: str
    ) -> HierarchyQueryResult:
        """Get operators under a module matching a pattern."""
        module_key = self._find_matching_key(module, self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_matching_pattern_under_module",
                title=f"Operators under Module '{module}' matching '{pattern}'",
                data=None,
                message=f"Module '{module}' not found."
            )

        operators = self.module_to_operators.get(module_key, set())
        matching = self._filter_by_pattern(operators, pattern)

        return HierarchyQueryResult(
            success=True,
            query_type="operators_matching_pattern_under_module",
            title=f"Operators under Module '{module_key}' matching '{pattern}'",
            data=sorted(matching),
            count=len(matching)
        )

    # =========================================================================
    # FROM SOURCE NAME QUERIES (6)
    # =========================================================================

    def get_vendors_count_under_source_name(self, source_name: str) -> HierarchyQueryResult:
        """Get number of vendors under a source name."""
        sn_key = self._find_matching_key(source_name, self.source_names)
        if not sn_key:
            return HierarchyQueryResult(
                success=False,
                query_type="vendors_count_under_source_name",
                title=f"Vendors under Source Name '{source_name}'",
                data=None,
                message=f"Source Name '{source_name}' not found. Available: {', '.join(sorted(list(self.source_names)[:10]))}..."
            )

        vendors = self.source_name_to_vendors.get(sn_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="vendors_count_under_source_name",
            title=f"Number of Vendors under Source Name '{sn_key}'",
            data=sorted(vendors),
            count=len(vendors)
        )

    def get_operators_count_under_source_name(self, source_name: str) -> HierarchyQueryResult:
        """Get number of operators under a source name."""
        sn_key = self._find_matching_key(source_name, self.source_names)
        if not sn_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_count_under_source_name",
                title=f"Operators under Source Name '{source_name}'",
                data=None,
                message=f"Source Name '{source_name}' not found."
            )

        operators = self.source_name_to_operators.get(sn_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="operators_count_under_source_name",
            title=f"Number of Operators under Source Name '{sn_key}'",
            data=sorted(operators),
            count=len(operators)
        )

    def get_vendors_list_under_source_name(self, source_name: str) -> HierarchyQueryResult:
        """Get list of vendors under a source name."""
        return self.get_vendors_count_under_source_name(source_name)

    def get_operators_list_under_source_name(self, source_name: str) -> HierarchyQueryResult:
        """Get list of operators under a source name."""
        return self.get_operators_count_under_source_name(source_name)

    def get_vendors_with_min_operators_under_source_name(
            self, source_name: str, min_operators: int
    ) -> HierarchyQueryResult:
        """Get vendors under a source name with more than X operators."""
        sn_key = self._find_matching_key(source_name, self.source_names)
        if not sn_key:
            return HierarchyQueryResult(
                success=False,
                query_type="vendors_with_min_operators_under_source_name",
                title=f"Vendors under Source Name '{source_name}' with >{min_operators} operators",
                data=None,
                message=f"Source Name '{source_name}' not found."
            )

        vendors = self.source_name_to_vendors.get(sn_key, set())

        matching_vendors = {}
        for vendor in vendors:
            vendor_operators = self.vendor_to_operators.get(vendor, set())
            sn_operators = self.source_name_to_operators.get(sn_key, set())
            common_operators = vendor_operators & sn_operators

            if len(common_operators) > min_operators:
                matching_vendors[vendor] = sorted(common_operators)

        return HierarchyQueryResult(
            success=True,
            query_type="vendors_with_min_operators_under_source_name",
            title=f"Vendors under Source Name '{sn_key}' with more than {min_operators} operators",
            data=matching_vendors,
            count=len(matching_vendors)
        )

    def get_operators_matching_pattern_under_source_name(
            self, source_name: str, pattern: str
    ) -> HierarchyQueryResult:
        """Get operators under a source name matching a pattern."""
        sn_key = self._find_matching_key(source_name, self.source_names)
        if not sn_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_matching_pattern_under_source_name",
                title=f"Operators under Source Name '{source_name}' matching '{pattern}'",
                data=None,
                message=f"Source Name '{source_name}' not found."
            )

        operators = self.source_name_to_operators.get(sn_key, set())
        matching = self._filter_by_pattern(operators, pattern)

        return HierarchyQueryResult(
            success=True,
            query_type="operators_matching_pattern_under_source_name",
            title=f"Operators under Source Name '{sn_key}' matching '{pattern}'",
            data=sorted(matching),
            count=len(matching)
        )

    # =========================================================================
    # FROM VENDOR QUERIES (3)
    # =========================================================================

    def get_operators_count_under_vendor(self, vendor: str) -> HierarchyQueryResult:
        """Get number of operators under a vendor."""
        vendor_key = self._find_matching_key(vendor, self.vendors)
        if not vendor_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_count_under_vendor",
                title=f"Operators under Vendor '{vendor}'",
                data=None,
                message=f"Vendor '{vendor}' not found. Available vendors: {', '.join(sorted(list(self.vendors)[:10]))}..."
            )

        operators = self.vendor_to_operators.get(vendor_key, set())
        return HierarchyQueryResult(
            success=True,
            query_type="operators_count_under_vendor",
            title=f"Number of Operators under Vendor '{vendor_key}'",
            data=sorted(operators),
            count=len(operators)
        )

    def get_operators_list_under_vendor(self, vendor: str) -> HierarchyQueryResult:
        """Get list of operators under a vendor."""
        return self.get_operators_count_under_vendor(vendor)

    def get_operators_matching_pattern_under_vendor(
            self, vendor: str, pattern: str
    ) -> HierarchyQueryResult:
        """Get operators under a vendor matching a pattern."""
        vendor_key = self._find_matching_key(vendor, self.vendors)
        if not vendor_key:
            return HierarchyQueryResult(
                success=False,
                query_type="operators_matching_pattern_under_vendor",
                title=f"Operators under Vendor '{vendor}' matching '{pattern}'",
                data=None,
                message=f"Vendor '{vendor}' not found."
            )

        operators = self.vendor_to_operators.get(vendor_key, set())
        matching = self._filter_by_pattern(operators, pattern)

        return HierarchyQueryResult(
            success=True,
            query_type="operators_matching_pattern_under_vendor",
            title=f"Operators under Vendor '{vendor_key}' matching '{pattern}'",
            data=sorted(matching),
            count=len(matching)
        )

    # =========================================================================
    # GLOBAL / SYSTEM-LEVEL QUERIES (16)
    # =========================================================================

    def get_total_sources(self) -> HierarchyQueryResult:
        """Get total number of sources."""
        return HierarchyQueryResult(
            success=True,
            query_type="total_sources",
            title="Total Number of Sources",
            data=sorted(self.sources),
            count=len(self.sources)
        )

    def get_total_modules(self) -> HierarchyQueryResult:
        """Get total number of modules."""
        return HierarchyQueryResult(
            success=True,
            query_type="total_modules",
            title="Total Number of Modules",
            data=sorted(self.modules),
            count=len(self.modules)
        )

    def get_total_source_names(self) -> HierarchyQueryResult:
        """Get total number of source names."""
        return HierarchyQueryResult(
            success=True,
            query_type="total_source_names",
            title="Total Number of Source Names",
            data=sorted(self.source_names),
            count=len(self.source_names)
        )

    def get_total_vendors(self) -> HierarchyQueryResult:
        """Get total number of vendors."""
        return HierarchyQueryResult(
            success=True,
            query_type="total_vendors",
            title="Total Number of Vendors",
            data=sorted(self.vendors),
            count=len(self.vendors)
        )

    def get_total_operators(self) -> HierarchyQueryResult:
        """Get total number of operators."""
        return HierarchyQueryResult(
            success=True,
            query_type="total_operators",
            title="Total Number of Operators",
            data=sorted(self.operators),
            count=len(self.operators)
        )

    def get_modules_grouped_by_source(self) -> HierarchyQueryResult:
        """Get number of modules grouped by source."""
        grouped = {
            source: sorted(modules)
            for source, modules in sorted(self.source_to_modules.items())
        }
        return HierarchyQueryResult(
            success=True,
            query_type="modules_grouped_by_source",
            title="Modules Grouped by Source",
            data=grouped,
            count=len(grouped)
        )

    def get_vendors_grouped_by_source(self) -> HierarchyQueryResult:
        """Get number of vendors grouped by source."""
        grouped = {
            source: sorted(vendors)
            for source, vendors in sorted(self.source_to_vendors.items())
        }
        return HierarchyQueryResult(
            success=True,
            query_type="vendors_grouped_by_source",
            title="Vendors Grouped by Source",
            data=grouped,
            count=len(grouped)
        )

    def get_operators_grouped_by_vendor(self) -> HierarchyQueryResult:
        """Get number of operators grouped by vendor."""
        grouped = {
            vendor: sorted(operators)
            for vendor, operators in sorted(self.vendor_to_operators.items())
            if operators  # Only include vendors with operators
        }
        return HierarchyQueryResult(
            success=True,
            query_type="operators_grouped_by_vendor",
            title="Operators Grouped by Vendor",
            data=grouped,
            count=len(grouped)
        )

    def get_top_vendors_by_operators(self, n: int = 10) -> HierarchyQueryResult:
        """Get top N vendors with most operators."""
        vendor_counts = [
            (vendor, len(operators), sorted(operators))
            for vendor, operators in self.vendor_to_operators.items()
            if operators
        ]
        vendor_counts.sort(key=lambda x: -x[1])
        top_n = vendor_counts[:n]

        return HierarchyQueryResult(
            success=True,
            query_type="top_vendors_by_operators",
            title=f"Top {n} Vendors by Operator Count",
            data=top_n,
            count=len(top_n)
        )

    def get_top_sources_by_modules(self, n: int = 10) -> HierarchyQueryResult:
        """Get top N sources with most modules."""
        source_counts = [
            (source, len(modules), sorted(modules))
            for source, modules in self.source_to_modules.items()
        ]
        source_counts.sort(key=lambda x: -x[1])
        top_n = source_counts[:n]

        return HierarchyQueryResult(
            success=True,
            query_type="top_sources_by_modules",
            title=f"Top {n} Sources by Module Count",
            data=top_n,
            count=len(top_n)
        )

    def get_modules_with_zero_source_names(self) -> HierarchyQueryResult:
        """Get modules with zero source names."""
        empty_modules = [
            module for module in self.modules
            if not self.module_to_source_names.get(module)
        ]
        return HierarchyQueryResult(
            success=True,
            query_type="modules_with_zero_source_names",
            title="Modules with Zero Source Names",
            data=sorted(empty_modules),
            count=len(empty_modules)
        )

    def get_source_names_with_zero_vendors(self) -> HierarchyQueryResult:
        """Get source names with zero vendors."""
        empty_source_names = [
            sn for sn in self.source_names
            if not self.source_name_to_vendors.get(sn)
        ]
        return HierarchyQueryResult(
            success=True,
            query_type="source_names_with_zero_vendors",
            title="Source Names with Zero Vendors",
            data=sorted(empty_source_names),
            count=len(empty_source_names)
        )

    def get_vendors_with_zero_operators(self) -> HierarchyQueryResult:
        """Get vendors with zero operators."""
        empty_vendors = [
            vendor for vendor in self.vendors
            if not self.vendor_to_operators.get(vendor)
        ]
        return HierarchyQueryResult(
            success=True,
            query_type="vendors_with_zero_operators",
            title="Vendors with Zero Operators",
            data=sorted(empty_vendors),
            count=len(empty_vendors)
        )

    def get_all_unique_source_names(self) -> HierarchyQueryResult:
        """Get all unique source names."""
        return HierarchyQueryResult(
            success=True,
            query_type="all_unique_source_names",
            title="All Unique Source Names",
            data=sorted(self.source_names),
            count=len(self.source_names)
        )

    def get_all_unique_vendor_names(self) -> HierarchyQueryResult:
        """Get all unique vendor names."""
        return HierarchyQueryResult(
            success=True,
            query_type="all_unique_vendor_names",
            title="All Unique Vendor Names",
            data=sorted(self.vendors),
            count=len(self.vendors)
        )

    def get_all_unique_operator_names(self) -> HierarchyQueryResult:
        """Get all unique operator names."""
        return HierarchyQueryResult(
            success=True,
            query_type="all_unique_operator_names",
            title="All Unique Operator Names",
            data=sorted(self.operators),
            count=len(self.operators)
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _find_matching_key(self, query: str, candidates: Set[str]) -> Optional[str]:
        """
        Find matching key using case-insensitive comparison.

        Args:
            query: User-provided value
            candidates: Set of valid values

        Returns:
            Matching key or None
        """
        query_lower = query.lower().strip()

        # Exact match first
        for candidate in candidates:
            if candidate.lower() == query_lower:
                return candidate

        # Partial match
        for candidate in candidates:
            if query_lower in candidate.lower() or candidate.lower() in query_lower:
                return candidate

        return None

    def _filter_by_pattern(self, items: Set[str], pattern: str) -> List[str]:
        """
        Filter items by pattern (supports * wildcard and regex).

        Args:
            items: Set of items to filter
            pattern: Pattern to match

        Returns:
            List of matching items
        """
        # Convert wildcard pattern to regex
        if '*' in pattern and not pattern.startswith('^'):
            regex_pattern = pattern.replace('*', '.*')
        else:
            regex_pattern = pattern

        try:
            compiled = re.compile(regex_pattern, re.IGNORECASE)
            return [item for item in items if compiled.search(item)]
        except re.error:
            # Fall back to simple substring match
            pattern_lower = pattern.lower().replace('*', '')
            return [item for item in items if pattern_lower in item.lower()]

    # =========================================================================
    # RESULT FORMATTING
    # =========================================================================

    def format_result(self, result: HierarchyQueryResult) -> str:
        """
        Format a query result for display.

        Args:
            result: HierarchyQueryResult to format

        Returns:
            Formatted string for display
        """
        if not result.success:
            return f"❌ **Error:** {result.message}"

        response = f"## {result.title}\n\n"

        if result.count is not None:
            response += f"**Count:** {result.count}\n\n"

        data = result.data

        # Format based on data type
        if isinstance(data, list):
            if len(data) == 0:
                response += "*No items found.*\n"
            elif isinstance(data[0], tuple) and len(data[0]) == 3:
                # Top N format: (name, count, items)
                for i, (name, count, items) in enumerate(data, 1):
                    response += f"{i}. **{name}** ({count})\n"
                    if items and len(items) <= 10:
                        response += f"   - {', '.join(items)}\n"
                    elif items:
                        response += f"   - {', '.join(items[:10])}... (+{len(items) - 10} more)\n"
            else:
                # Simple list
                if len(data) <= 20:
                    for item in data:
                        response += f"- {item}\n"
                else:
                    for item in data[:20]:
                        response += f"- {item}\n"
                    response += f"\n*...and {len(data) - 20} more*\n"

        elif isinstance(data, dict):
            if len(data) == 0:
                response += "*No items found.*\n"
            else:
                for key, value in list(data.items())[:20]:
                    if isinstance(value, list):
                        response += f"### {key} ({len(value)})\n"
                        if len(value) <= 10:
                            response += f"- {', '.join(value)}\n"
                        else:
                            response += f"- {', '.join(value[:10])}... (+{len(value) - 10} more)\n"
                        response += "\n"
                    else:
                        response += f"- **{key}:** {value}\n"

                if len(data) > 20:
                    response += f"\n*...and {len(data) - 20} more entries*\n"

        return response