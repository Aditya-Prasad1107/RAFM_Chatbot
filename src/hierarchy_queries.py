"""
Hierarchy Query Engine for Mapping ChatBot.
Provides 45+ query capabilities for navigating the Domain/Module/Source/Vendor/Operator hierarchy.

Query Categories:
1. From Domain (12 queries)
2. From Module (9 queries)
3. From Source (6 queries)
4. From Vendor (3 queries)
5. Global/System-Level (15 queries)

Note: All hierarchy values (Domain, Module, Source, Vendor, Operator) are stored in UPPERCASE.
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

    Hierarchy: Domain → Module → Source → Vendor → Operator
    All values stored in UPPERCASE.
    """

    def __init__(self, chatbot_instance):
        """Initialize with reference to the chatbot."""
        self.chatbot = chatbot_instance
        self._build_indexes()

    def _build_indexes(self):
        """Build inverted indexes for fast hierarchy queries."""
        self.domains: Set[str] = set()
        self.modules: Set[str] = set()
        self.sources: Set[str] = set()
        self.vendors: Set[str] = set()
        self.operators: Set[str] = set()

        self.domain_to_modules: Dict[str, Set[str]] = defaultdict(set)
        self.domain_to_sources: Dict[str, Set[str]] = defaultdict(set)
        self.domain_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.domain_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.module_to_sources: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.source_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.source_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.vendor_to_operators: Dict[str, Set[str]] = defaultdict(set)

        self.operator_to_vendors: Dict[str, Set[str]] = defaultdict(set)
        self.vendor_to_sources: Dict[str, Set[str]] = defaultdict(set)
        self.source_to_modules: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_domains: Dict[str, Set[str]] = defaultdict(set)

        self.full_paths: List[Tuple[str, str, str, str]] = []

        for meta in self.chatbot.metadata:
            domain = meta['domain'].upper()
            module = meta['module'].upper()
            source = meta['source'].upper()
            vendor = meta['vendor'].upper()

            self.domains.add(domain)
            self.modules.add(module)
            self.sources.add(source)
            self.vendors.add(vendor)

            self.domain_to_modules[domain].add(module)
            self.domain_to_sources[domain].add(source)
            self.domain_to_vendors[domain].add(vendor)

            self.module_to_sources[module].add(source)
            self.module_to_vendors[module].add(vendor)

            self.source_to_vendors[source].add(vendor)

            self.module_to_domains[module].add(domain)
            self.source_to_modules[source].add(module)
            self.vendor_to_sources[vendor].add(source)

            self.full_paths.append((domain, module, source, vendor))

        self._build_operator_indexes()

    def _build_operator_indexes(self):
        """Build operator indexes by extracting from filenames."""
        for key, expr_filenames in self.chatbot.expression_filenames_data.items():
            domain, module, source, vendor = key
            domain = domain.upper()
            module = module.upper()
            source = source.upper()
            vendor = vendor.upper()

            unique_filenames = set()
            for filenames in expr_filenames.values():
                unique_filenames.update(filenames)

            if key in self.chatbot.filename_data:
                for filenames in self.chatbot.filename_data[key].values():
                    unique_filenames.update(filenames)

            for filename in unique_filenames:
                operator = self.chatbot.extract_operator_from_filename(
                    filename, source, vendor
                )
                if operator and operator != "Unknown":
                    operator = operator.upper()
                    self.operators.add(operator)
                    self.domain_to_operators[domain].add(operator)
                    self.module_to_operators[module].add(operator)
                    self.source_to_operators[source].add(operator)
                    self.vendor_to_operators[vendor].add(operator)
                    self.operator_to_vendors[operator].add(vendor)

    # =========================================================================
    # FROM DOMAIN QUERIES (12)
    # =========================================================================

    def get_modules_count_under_domain(self, domain: str) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="modules_count_under_domain",
                title=f"Modules under Domain '{domain.upper()}'", data=None,
                message=f"Domain '{domain.upper()}' not found. Available: {', '.join(sorted(self.domains))}"
            )
        modules = self.domain_to_modules.get(domain_key, set())
        return HierarchyQueryResult(
            success=True, query_type="modules_count_under_domain",
            title=f"Modules under Domain '{domain_key}'",
            data=sorted(modules), count=len(modules)
        )

    def get_sources_count_under_domain(self, domain: str) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="sources_count_under_domain",
                title=f"Sources under Domain '{domain.upper()}'", data=None,
                message=f"Domain '{domain.upper()}' not found."
            )
        sources = self.domain_to_sources.get(domain_key, set())
        return HierarchyQueryResult(
            success=True, query_type="sources_count_under_domain",
            title=f"Sources under Domain '{domain_key}'",
            data=sorted(sources), count=len(sources)
        )

    def get_vendors_count_under_domain(self, domain: str) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="vendors_count_under_domain",
                title=f"Vendors under Domain '{domain.upper()}'", data=None,
                message=f"Domain '{domain.upper()}' not found."
            )
        vendors = self.domain_to_vendors.get(domain_key, set())
        return HierarchyQueryResult(
            success=True, query_type="vendors_count_under_domain",
            title=f"Vendors under Domain '{domain_key}'",
            data=sorted(vendors), count=len(vendors)
        )

    def get_operators_count_under_domain(self, domain: str) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_count_under_domain",
                title=f"Operators under Domain '{domain.upper()}'", data=None,
                message=f"Domain '{domain.upper()}' not found."
            )
        operators = self.domain_to_operators.get(domain_key, set())
        return HierarchyQueryResult(
            success=True, query_type="operators_count_under_domain",
            title=f"Operators under Domain '{domain_key}'",
            data=sorted(operators), count=len(operators)
        )

    def get_modules_list_under_domain(self, domain: str) -> HierarchyQueryResult:
        return self.get_modules_count_under_domain(domain)

    def get_sources_list_under_domain(self, domain: str) -> HierarchyQueryResult:
        return self.get_sources_count_under_domain(domain)

    def get_vendors_list_under_domain(self, domain: str) -> HierarchyQueryResult:
        return self.get_vendors_count_under_domain(domain)

    def get_operators_list_under_domain(self, domain: str) -> HierarchyQueryResult:
        return self.get_operators_count_under_domain(domain)

    def get_vendors_with_min_operators_under_domain(self, domain: str, min_operators: int) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="vendors_with_min_operators_under_domain",
                title=f"Vendors under Domain '{domain.upper()}' with >{min_operators} operators",
                data=None, message=f"Domain '{domain.upper()}' not found."
            )
        vendors = self.domain_to_vendors.get(domain_key, set())
        matching_vendors = {}
        for vendor in vendors:
            vendor_operators = self.vendor_to_operators.get(vendor, set())
            domain_operators = self.domain_to_operators.get(domain_key, set())
            common_operators = vendor_operators & domain_operators
            if len(common_operators) > min_operators:
                matching_vendors[vendor] = sorted(common_operators)
        return HierarchyQueryResult(
            success=True, query_type="vendors_with_min_operators_under_domain",
            title=f"Vendors under Domain '{domain_key}' with more than {min_operators} operators",
            data=matching_vendors, count=len(matching_vendors)
        )

    def get_operators_matching_pattern_under_domain(self, domain: str, pattern: str) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_matching_pattern_under_domain",
                title=f"Operators under Domain '{domain.upper()}' matching '{pattern}'",
                data=None, message=f"Domain '{domain.upper()}' not found."
            )
        operators = self.domain_to_operators.get(domain_key, set())
        matching = self._filter_by_pattern(operators, pattern.upper())
        return HierarchyQueryResult(
            success=True, query_type="operators_matching_pattern_under_domain",
            title=f"Operators under Domain '{domain_key}' matching '{pattern.upper()}'",
            data=sorted(matching), count=len(matching)
        )

    def get_sources_containing_keyword_under_domain(self, domain: str, keyword: str) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="sources_containing_keyword_under_domain",
                title=f"Sources under Domain '{domain.upper()}' containing '{keyword}'",
                data=None, message=f"Domain '{domain.upper()}' not found."
            )
        sources = self.domain_to_sources.get(domain_key, set())
        matching = [s for s in sources if keyword.upper() in s.upper()]
        return HierarchyQueryResult(
            success=True, query_type="sources_containing_keyword_under_domain",
            title=f"Sources under Domain '{domain_key}' containing '{keyword.upper()}'",
            data=sorted(matching), count=len(matching)
        )

    def get_modules_with_min_sources_under_domain(self, domain: str, min_sources: int) -> HierarchyQueryResult:
        domain_key = self._find_matching_key(domain.upper(), self.domains)
        if not domain_key:
            return HierarchyQueryResult(
                success=False, query_type="modules_with_min_sources_under_domain",
                title=f"Modules under Domain '{domain.upper()}' with >={min_sources} sources",
                data=None, message=f"Domain '{domain.upper()}' not found."
            )
        modules = self.domain_to_modules.get(domain_key, set())
        matching_modules = {}
        for module in modules:
            module_sources = self.module_to_sources.get(module, set())
            domain_sources = self.domain_to_sources.get(domain_key, set())
            common = module_sources & domain_sources
            if len(common) >= min_sources:
                matching_modules[module] = sorted(common)
        return HierarchyQueryResult(
            success=True, query_type="modules_with_min_sources_under_domain",
            title=f"Modules under Domain '{domain_key}' with at least {min_sources} sources",
            data=matching_modules, count=len(matching_modules)
        )

    # =========================================================================
    # FROM MODULE QUERIES (9)
    # =========================================================================

    def get_sources_count_under_module(self, module: str) -> HierarchyQueryResult:
        module_key = self._find_matching_key(module.upper(), self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False, query_type="sources_count_under_module",
                title=f"Sources under Module '{module.upper()}'", data=None,
                message=f"Module '{module.upper()}' not found. Available: {', '.join(sorted(self.modules))}"
            )
        sources = self.module_to_sources.get(module_key, set())
        return HierarchyQueryResult(
            success=True, query_type="sources_count_under_module",
            title=f"Sources under Module '{module_key}'",
            data=sorted(sources), count=len(sources)
        )

    def get_vendors_count_under_module(self, module: str) -> HierarchyQueryResult:
        module_key = self._find_matching_key(module.upper(), self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False, query_type="vendors_count_under_module",
                title=f"Vendors under Module '{module.upper()}'", data=None,
                message=f"Module '{module.upper()}' not found."
            )
        vendors = self.module_to_vendors.get(module_key, set())
        return HierarchyQueryResult(
            success=True, query_type="vendors_count_under_module",
            title=f"Vendors under Module '{module_key}'",
            data=sorted(vendors), count=len(vendors)
        )

    def get_operators_count_under_module(self, module: str) -> HierarchyQueryResult:
        module_key = self._find_matching_key(module.upper(), self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_count_under_module",
                title=f"Operators under Module '{module.upper()}'", data=None,
                message=f"Module '{module.upper()}' not found."
            )
        operators = self.module_to_operators.get(module_key, set())
        return HierarchyQueryResult(
            success=True, query_type="operators_count_under_module",
            title=f"Operators under Module '{module_key}'",
            data=sorted(operators), count=len(operators)
        )

    def get_sources_list_under_module(self, module: str) -> HierarchyQueryResult:
        return self.get_sources_count_under_module(module)

    def get_vendors_list_under_module(self, module: str) -> HierarchyQueryResult:
        return self.get_vendors_count_under_module(module)

    def get_operators_list_under_module(self, module: str) -> HierarchyQueryResult:
        return self.get_operators_count_under_module(module)

    def get_sources_containing_keyword_under_module(self, module: str, keyword: str) -> HierarchyQueryResult:
        module_key = self._find_matching_key(module.upper(), self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False, query_type="sources_containing_keyword_under_module",
                title=f"Sources under Module '{module.upper()}' containing '{keyword}'",
                data=None, message=f"Module '{module.upper()}' not found."
            )
        sources = self.module_to_sources.get(module_key, set())
        matching = [s for s in sources if keyword.upper() in s.upper()]
        return HierarchyQueryResult(
            success=True, query_type="sources_containing_keyword_under_module",
            title=f"Sources under Module '{module_key}' containing '{keyword.upper()}'",
            data=sorted(matching), count=len(matching)
        )

    def get_vendors_with_min_operators_under_module(self, module: str, min_operators: int) -> HierarchyQueryResult:
        module_key = self._find_matching_key(module.upper(), self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False, query_type="vendors_with_min_operators_under_module",
                title=f"Vendors under Module '{module.upper()}' with >{min_operators} operators",
                data=None, message=f"Module '{module.upper()}' not found."
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
            success=True, query_type="vendors_with_min_operators_under_module",
            title=f"Vendors under Module '{module_key}' with more than {min_operators} operators",
            data=matching_vendors, count=len(matching_vendors)
        )

    def get_operators_matching_pattern_under_module(self, module: str, pattern: str) -> HierarchyQueryResult:
        module_key = self._find_matching_key(module.upper(), self.modules)
        if not module_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_matching_pattern_under_module",
                title=f"Operators under Module '{module.upper()}' matching '{pattern}'",
                data=None, message=f"Module '{module.upper()}' not found."
            )
        operators = self.module_to_operators.get(module_key, set())
        matching = self._filter_by_pattern(operators, pattern.upper())
        return HierarchyQueryResult(
            success=True, query_type="operators_matching_pattern_under_module",
            title=f"Operators under Module '{module_key}' matching '{pattern.upper()}'",
            data=sorted(matching), count=len(matching)
        )

    # =========================================================================
    # FROM SOURCE QUERIES (6)
    # =========================================================================

    def get_vendors_count_under_source(self, source: str) -> HierarchyQueryResult:
        source_key = self._find_matching_key(source.upper(), self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False, query_type="vendors_count_under_source",
                title=f"Vendors under Source '{source.upper()}'", data=None,
                message=f"Source '{source.upper()}' not found."
            )
        vendors = self.source_to_vendors.get(source_key, set())
        return HierarchyQueryResult(
            success=True, query_type="vendors_count_under_source",
            title=f"Vendors under Source '{source_key}'",
            data=sorted(vendors), count=len(vendors)
        )

    def get_operators_count_under_source(self, source: str) -> HierarchyQueryResult:
        source_key = self._find_matching_key(source.upper(), self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_count_under_source",
                title=f"Operators under Source '{source.upper()}'", data=None,
                message=f"Source '{source.upper()}' not found."
            )
        operators = self.source_to_operators.get(source_key, set())
        return HierarchyQueryResult(
            success=True, query_type="operators_count_under_source",
            title=f"Operators under Source '{source_key}'",
            data=sorted(operators), count=len(operators)
        )

    def get_vendors_list_under_source(self, source: str) -> HierarchyQueryResult:
        return self.get_vendors_count_under_source(source)

    def get_operators_list_under_source(self, source: str) -> HierarchyQueryResult:
        return self.get_operators_count_under_source(source)

    def get_vendors_with_min_operators_under_source(self, source: str, min_operators: int) -> HierarchyQueryResult:
        source_key = self._find_matching_key(source.upper(), self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False, query_type="vendors_with_min_operators_under_source",
                title=f"Vendors under Source '{source.upper()}' with >{min_operators} operators",
                data=None, message=f"Source '{source.upper()}' not found."
            )
        vendors = self.source_to_vendors.get(source_key, set())
        matching_vendors = {}
        for vendor in vendors:
            vendor_operators = self.vendor_to_operators.get(vendor, set())
            source_operators = self.source_to_operators.get(source_key, set())
            common_operators = vendor_operators & source_operators
            if len(common_operators) > min_operators:
                matching_vendors[vendor] = sorted(common_operators)
        return HierarchyQueryResult(
            success=True, query_type="vendors_with_min_operators_under_source",
            title=f"Vendors under Source '{source_key}' with more than {min_operators} operators",
            data=matching_vendors, count=len(matching_vendors)
        )

    def get_operators_matching_pattern_under_source(self, source: str, pattern: str) -> HierarchyQueryResult:
        source_key = self._find_matching_key(source.upper(), self.sources)
        if not source_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_matching_pattern_under_source",
                title=f"Operators under Source '{source.upper()}' matching '{pattern}'",
                data=None, message=f"Source '{source.upper()}' not found."
            )
        operators = self.source_to_operators.get(source_key, set())
        matching = self._filter_by_pattern(operators, pattern.upper())
        return HierarchyQueryResult(
            success=True, query_type="operators_matching_pattern_under_source",
            title=f"Operators under Source '{source_key}' matching '{pattern.upper()}'",
            data=sorted(matching), count=len(matching)
        )

    # =========================================================================
    # FROM VENDOR QUERIES (3)
    # =========================================================================

    def get_operators_count_under_vendor(self, vendor: str) -> HierarchyQueryResult:
        vendor_key = self._find_matching_key(vendor.upper(), self.vendors)
        if not vendor_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_count_under_vendor",
                title=f"Operators under Vendor '{vendor.upper()}'", data=None,
                message=f"Vendor '{vendor.upper()}' not found."
            )
        operators = self.vendor_to_operators.get(vendor_key, set())
        return HierarchyQueryResult(
            success=True, query_type="operators_count_under_vendor",
            title=f"Operators under Vendor '{vendor_key}'",
            data=sorted(operators), count=len(operators)
        )

    def get_operators_list_under_vendor(self, vendor: str) -> HierarchyQueryResult:
        return self.get_operators_count_under_vendor(vendor)

    def get_operators_matching_pattern_under_vendor(self, vendor: str, pattern: str) -> HierarchyQueryResult:
        vendor_key = self._find_matching_key(vendor.upper(), self.vendors)
        if not vendor_key:
            return HierarchyQueryResult(
                success=False, query_type="operators_matching_pattern_under_vendor",
                title=f"Operators under Vendor '{vendor.upper()}' matching '{pattern}'",
                data=None, message=f"Vendor '{vendor.upper()}' not found."
            )
        operators = self.vendor_to_operators.get(vendor_key, set())
        matching = self._filter_by_pattern(operators, pattern.upper())
        return HierarchyQueryResult(
            success=True, query_type="operators_matching_pattern_under_vendor",
            title=f"Operators under Vendor '{vendor_key}' matching '{pattern.upper()}'",
            data=sorted(matching), count=len(matching)
        )

    # =========================================================================
    # GLOBAL / SYSTEM-LEVEL QUERIES (16)
    # =========================================================================

    def get_total_domains(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="total_domains",
            title="Total Number of Domains",
            data=sorted(self.domains), count=len(self.domains)
        )

    def get_total_modules(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="total_modules",
            title="Total Number of Modules",
            data=sorted(self.modules), count=len(self.modules)
        )

    def get_total_sources(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="total_sources",
            title="Total Number of Sources",
            data=sorted(self.sources), count=len(self.sources)
        )

    def get_total_vendors(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="total_vendors",
            title="Total Number of Vendors",
            data=sorted(self.vendors), count=len(self.vendors)
        )

    def get_total_operators(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="total_operators",
            title="Total Number of Operators",
            data=sorted(self.operators), count=len(self.operators)
        )

    def get_modules_grouped_by_domain(self) -> HierarchyQueryResult:
        grouped = {
            domain: sorted(modules)
            for domain, modules in sorted(self.domain_to_modules.items())
        }
        return HierarchyQueryResult(
            success=True, query_type="modules_grouped_by_domain",
            title="Modules Grouped by Domain",
            data=grouped, count=len(grouped)
        )

    def get_vendors_grouped_by_domain(self) -> HierarchyQueryResult:
        grouped = {
            domain: sorted(vendors)
            for domain, vendors in sorted(self.domain_to_vendors.items())
        }
        return HierarchyQueryResult(
            success=True, query_type="vendors_grouped_by_domain",
            title="Vendors Grouped by Domain",
            data=grouped, count=len(grouped)
        )

    def get_operators_grouped_by_vendor(self) -> HierarchyQueryResult:
        grouped = {
            vendor: sorted(operators)
            for vendor, operators in sorted(self.vendor_to_operators.items())
            if operators
        }
        return HierarchyQueryResult(
            success=True, query_type="operators_grouped_by_vendor",
            title="Operators Grouped by Vendor",
            data=grouped, count=len(grouped)
        )

    def get_top_vendors_by_operators(self, n: int = 10) -> HierarchyQueryResult:
        vendor_counts = [
            (vendor, len(operators), sorted(operators))
            for vendor, operators in self.vendor_to_operators.items()
            if operators
        ]
        vendor_counts.sort(key=lambda x: -x[1])
        top_n = vendor_counts[:n]
        return HierarchyQueryResult(
            success=True, query_type="top_vendors_by_operators",
            title=f"Top {n} Vendors by Operator Count",
            data=top_n, count=len(top_n)
        )

    def get_top_domains_by_modules(self, n: int = 10) -> HierarchyQueryResult:
        domain_counts = [
            (domain, len(modules), sorted(modules))
            for domain, modules in self.domain_to_modules.items()
        ]
        domain_counts.sort(key=lambda x: -x[1])
        top_n = domain_counts[:n]
        return HierarchyQueryResult(
            success=True, query_type="top_domains_by_modules",
            title=f"Top {n} Domains by Module Count",
            data=top_n, count=len(top_n)
        )

    def get_modules_with_zero_sources(self) -> HierarchyQueryResult:
        empty_modules = [
            module for module in self.modules
            if not self.module_to_sources.get(module)
        ]
        return HierarchyQueryResult(
            success=True, query_type="modules_with_zero_sources",
            title="Modules with Zero Sources",
            data=sorted(empty_modules), count=len(empty_modules)
        )

    def get_sources_with_zero_vendors(self) -> HierarchyQueryResult:
        empty_sources = [
            source for source in self.sources
            if not self.source_to_vendors.get(source)
        ]
        return HierarchyQueryResult(
            success=True, query_type="sources_with_zero_vendors",
            title="Sources with Zero Vendors",
            data=sorted(empty_sources), count=len(empty_sources)
        )

    def get_vendors_with_zero_operators(self) -> HierarchyQueryResult:
        empty_vendors = [
            vendor for vendor in self.vendors
            if not self.vendor_to_operators.get(vendor)
        ]
        return HierarchyQueryResult(
            success=True, query_type="vendors_with_zero_operators",
            title="Vendors with Zero Operators",
            data=sorted(empty_vendors), count=len(empty_vendors)
        )

    def get_all_unique_sources(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="all_unique_sources",
            title="All Unique Sources",
            data=sorted(self.sources), count=len(self.sources)
        )

    def get_all_unique_vendor_names(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="all_unique_vendor_names",
            title="All Unique Vendor Names",
            data=sorted(self.vendors), count=len(self.vendors)
        )

    def get_all_unique_operator_names(self) -> HierarchyQueryResult:
        return HierarchyQueryResult(
            success=True, query_type="all_unique_operator_names",
            title="All Unique Operator Names",
            data=sorted(self.operators), count=len(self.operators)
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _find_matching_key(self, query: str, candidates: Set[str]) -> Optional[str]:
        query_upper = query.upper().strip()
        if query_upper in candidates:
            return query_upper
        for candidate in candidates:
            if query_upper in candidate or candidate in query_upper:
                return candidate
        return None

    def _filter_by_pattern(self, items: Set[str], pattern: str) -> List[str]:
        if '*' in pattern and not pattern.startswith('^'):
            regex_pattern = pattern.replace('*', '.*')
        else:
            regex_pattern = pattern
        try:
            compiled = re.compile(regex_pattern, re.IGNORECASE)
            return [item for item in items if compiled.search(item)]
        except re.error:
            pattern_clean = pattern.replace('*', '')
            return [item for item in items if pattern_clean in item]

    # =========================================================================
    # RESULT FORMATTING - NO TRUNCATION (SHOW ALL)
    # =========================================================================

    def format_result(self, result: HierarchyQueryResult) -> str:
        """Format a query result for display - shows ALL items (no truncation)."""
        if not result.success:
            return f"❌ **Error:** {result.message}"

        response = f"## {result.title}\n\n"

        if result.count is not None:
            response += f"**Count:** {result.count}\n\n"

        data = result.data

        if isinstance(data, list):
            if len(data) == 0:
                response += "*No items found.*\n"
            elif isinstance(data[0], tuple) and len(data[0]) == 3:
                # Top N format: (name, count, items)
                for i, (name, count, items) in enumerate(data, 1):
                    response += f"{i}. **{name}** ({count})\n"
                    if items:
                        response += f"   - {', '.join(items)}\n"
            else:
                # Simple list - show ALL items
                for item in data:
                    response += f"- {item}\n"

        elif isinstance(data, dict):
            if len(data) == 0:
                response += "*No items found.*\n"
            else:
                # Show ALL entries
                for key, value in sorted(data.items()):
                    if isinstance(value, list):
                        response += f"### {key} ({len(value)})\n"
                        response += f"- {', '.join(value)}\n\n"
                    else:
                        response += f"- **{key}:** {value}\n"

        return response