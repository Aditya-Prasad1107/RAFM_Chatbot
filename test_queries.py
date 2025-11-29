"""
RAFM Chatbot - Automated Test Runner
=====================================

This script runs all 63+ query combinations against the chatbot and generates a test report.

Usage:
    1. Update ROOT_FOLDER path below to your data folder
    2. Run: python test_runner.py
    3. View the test report in the console and in test_report.txt
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chatbot import MappingChatBot

# ============================================================================
# CONFIGURATION - UPDATE THIS PATH
# ============================================================================

ROOT_FOLDER = r"C:\Users\aditya.prasad\OneDrive - Mobileum\Documents\OneDrive - Mobileum\Tejas N's files - Templates"

# Alternative: Use environment variable
# ROOT_FOLDER = os.getenv("MAPPING_ROOT_FOLDER", "./Source")

# ============================================================================
# TEST QUERIES
# ============================================================================

# Section 1: Single Filter Queries (6)
SINGLE_FILTER_QUERIES = [
    ("Field only", "get me logics for 'event_type'"),
    ("Source only", "show all mappings where source is RA"),
    ("Module only", "find mappings in module UC"),
    ("Source Name only", "show mappings for source name MSC"),
    ("Vendor only", "list all mappings for vendor Nokia"),
    ("Operator only", "show mappings where operator is DU"),
]

# Section 2: Two Filter Combinations (15)
TWO_FILTER_QUERIES = [
    ("Field + Source", "get 'event_type' where source is RA"),
    ("Field + Module", "get 'event_type' where module is UC"),
    ("Field + Source Name", "get 'event_type' where source name is MSC"),
    ("Field + Vendor", "get 'event_type' where vendor is Nokia"),
    ("Field + Operator", "get 'event_type' where operator is DU"),
    ("Source + Module", "show mappings where source is RA and module is UC"),
    ("Source + Source Name", "show mappings where source is RA and source name is MSC"),
    ("Source + Vendor", "show mappings where source is RA and vendor is Nokia"),
    ("Source + Operator", "show mappings where source is RA and operator is DU"),
    ("Module + Source Name", "show mappings where module is UC and source name is MSC"),
    ("Module + Vendor", "show mappings where module is UC and vendor is Nokia"),
    ("Module + Operator", "show mappings where module is UC and operator is DU"),
    ("Source Name + Vendor", "show mappings where source name is MSC and vendor is Nokia"),
    ("Source Name + Operator", "show mappings where source name is MSC and operator is DU"),
    ("Vendor + Operator", "show mappings where vendor is Nokia and operator is DU"),
]

# Section 3: Three Filter Combinations (20)
THREE_FILTER_QUERIES = [
    ("Field + Source + Module", "get 'event_type' where source is RA and module is UC"),
    ("Field + Source + Source Name", "get 'event_type' where source is RA and source name is MSC"),
    ("Field + Source + Vendor", "get 'event_type' where source is RA and vendor is Nokia"),
    ("Field + Source + Operator", "get 'event_type' where source is RA and operator is DU"),
    ("Field + Module + Source Name", "get 'event_type' where module is UC and source name is MSC"),
    ("Field + Module + Vendor", "get 'event_type' where module is UC and vendor is Nokia"),
    ("Field + Module + Operator", "get 'event_type' where module is UC and operator is DU"),
    ("Field + Source Name + Vendor", "get 'event_type' where source name is MSC and vendor is Nokia"),
    ("Field + Source Name + Operator", "get 'event_type' where source name is MSC and operator is DU"),
    ("Field + Vendor + Operator", "get 'event_type' where vendor is Nokia and operator is DU"),
    ("Source + Module + Source Name", "show mappings where source is RA, module is UC and source name is MSC"),
    ("Source + Module + Vendor", "show mappings where source is RA, module is UC and vendor is Nokia"),
    ("Source + Module + Operator", "show mappings where source is RA, module is UC and operator is DU"),
    ("Source + Source Name + Vendor", "show mappings where source is RA, source name is MSC and vendor is Nokia"),
    ("Source + Source Name + Operator", "show mappings where source is RA, source name is MSC and operator is DU"),
    ("Source + Vendor + Operator", "show mappings where source is RA, vendor is Nokia and operator is DU"),
    ("Module + Source Name + Vendor", "show mappings where module is UC, source name is MSC and vendor is Nokia"),
    ("Module + Source Name + Operator", "show mappings where module is UC, source name is MSC and operator is DU"),
    ("Module + Vendor + Operator", "show mappings where module is UC, vendor is Nokia and operator is DU"),
    ("Source Name + Vendor + Operator", "show mappings where source name is MSC, vendor is Nokia and operator is DU"),
]

# Section 4: Four Filter Combinations (15)
FOUR_FILTER_QUERIES = [
    ("Field + Source + Module + Source Name", "get 'event_type' where source is RA, module is UC and source name is MSC"),
    ("Field + Source + Module + Vendor", "get 'event_type' where source is RA, module is UC and vendor is Nokia"),
    ("Field + Source + Module + Operator", "get 'event_type' where source is RA, module is UC and operator is DU"),
    ("Field + Source + Source Name + Vendor", "get 'event_type' where source is RA, source name is MSC and vendor is Nokia"),
    ("Field + Source + Source Name + Operator", "get 'event_type' where source is RA, source name is MSC and operator is DU"),
    ("Field + Source + Vendor + Operator", "get 'event_type' where source is RA, vendor is Nokia and operator is DU"),
    ("Field + Module + Source Name + Vendor", "get 'event_type' where module is UC, source name is MSC and vendor is Nokia"),
    ("Field + Module + Source Name + Operator", "get 'event_type' where module is UC, source name is MSC and operator is DU"),
    ("Field + Module + Vendor + Operator", "get 'event_type' where module is UC, vendor is Nokia and operator is DU"),
    ("Field + Source Name + Vendor + Operator", "get 'event_type' where source name is MSC, vendor is Nokia and operator is DU"),
    ("Source + Module + Source Name + Vendor", "show mappings where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Source + Module + Source Name + Operator", "show mappings where source is RA, module is UC, source name is MSC and operator is DU"),
    ("Source + Module + Vendor + Operator", "show mappings where source is RA, module is UC, vendor is Nokia and operator is DU"),
    ("Source + Source Name + Vendor + Operator", "show mappings where source is RA, source name is MSC, vendor is Nokia and operator is DU"),
    ("Module + Source Name + Vendor + Operator", "show mappings where module is UC, source name is MSC, vendor is Nokia and operator is DU"),
]

# Section 5: Five Filter Combinations (6)
FIVE_FILTER_QUERIES = [
    ("Field + Source + Module + Source Name + Vendor", "get 'event_type' where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Field + Source + Module + Source Name + Operator", "get 'event_type' where source is RA, module is UC, source name is MSC and operator is DU"),
    ("Field + Source + Module + Vendor + Operator", "get 'event_type' where source is RA, module is UC, vendor is Nokia and operator is DU"),
    ("Field + Source + Source Name + Vendor + Operator", "get 'event_type' where source is RA, source name is MSC, vendor is Nokia and operator is DU"),
    ("Field + Module + Source Name + Vendor + Operator", "get 'event_type' where module is UC, source name is MSC, vendor is Nokia and operator is DU"),
    ("Source + Module + Source Name + Vendor + Operator", "show mappings where source is RA, module is UC, source name is MSC, vendor is Nokia and operator is DU"),
]

# Section 6: All Six Filters (1)
ALL_FILTER_QUERIES = [
    ("All 6 Filters", "get 'event_type' where source is RA, module is UC, source name is MSC, vendor is Nokia and operator is DU"),
]

# Section 7: Alternative Phrasing Styles (12)
ALTERNATIVE_PHRASING_QUERIES = [
    ("Style: get me logics", "get me logics for 'event_type' where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Style: show mapping", "show mapping for 'event_type' where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Style: find", "find 'event_type' where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Style: what is", "what is the mapping for 'event_type' where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Style: from keyword", "get 'event_type' from source RA, module UC, source name MSC, vendor Nokia"),
    ("Style: in for module", "get 'event_type' in module UC where source is RA, source name is MSC and vendor is Nokia"),
    ("Style: unquoted field", "get event_type where source is RA, module is UC, source name is MSC and vendor is Nokia"),
    ("Style: double quotes", 'get "event_type" where source is RA, module is UC, source name is MSC and vendor is Nokia'),
    ("Style: for keyword", "mapping for field 'event_type' source RA module UC source name MSC vendor Nokia"),
    ("Style: question format", "what is the logic for event_type in RA, UC, MSC, Nokia?"),
    ("Style: my prefix", "show mappings where my source is RA and my module is UC and my vendor is Nokia"),
    ("Style: compact", "event_type source RA module UC source name MSC vendor Nokia operator DU"),
]

# Section 8: Special Commands (16)
SPECIAL_COMMANDS = [
    ("Help (help)", "help"),
    ("Help (h)", "h"),
    ("Help (?)", "?"),
    ("List (list)", "list"),
    ("List (sources)", "sources"),
    ("List (show all)", "show all"),
    ("List (list all)", "list all"),
    ("Stats (stats)", "stats"),
    ("Stats (statistics)", "statistics"),
    ("Stats (info)", "info"),
    ("Cache (cache stats)", "cache stats"),
    ("Cache (cache)", "cache"),
    ("Cache (cachestats)", "cachestats"),
    ("Clear Cache (clear cache)", "clear cache"),
    ("Clear Cache (clearcache)", "clearcache"),
    ("Clear Cache (reset cache)", "reset cache"),
]

# Section 9: Edge Cases (14)
EDGE_CASE_QUERIES = [
    ("Case insensitive - upper", "get 'EVENT_TYPE' where source is ra, module is uc and vendor is NOKIA"),
    ("Case insensitive - mixed", "GET 'event_type' WHERE SOURCE IS RA AND MODULE IS UC"),
    ("Partial match - field", "get 'event' where source is RA"),
    ("Partial match - vendor", "show mappings vendor Nok"),
    ("Multiple vendors - or", "get 'event_type' where vendor is Nokia or Ericsson"),
    ("Multiple vendors - and", "show mappings where vendor is Nokia and Huawei"),
    ("Field with space", "get 'event type' where source is RA"),
    ("Field no separator", "get 'eventtype' where source is RA"),
    ("Minimal - just field", "event_type"),
    ("Minimal - quoted field", "'event_type'"),
    ("With dimension D", "get 'event_type' dimension D where source is RA"),
    ("Dimension + field", "show dimension D field 'event_type' vendor Nokia"),
    ("Natural language 1", "I need the mapping for event_type from RA source"),
    ("Natural language 2", "Can you show me event_type logic for Nokia vendor?"),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class TestRunner:
    """Automated test runner for RAFM Chatbot."""

    def __init__(self, root_folder: str):
        self.root_folder = root_folder
        self.chatbot = None
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'sections': {},
            'failed_queries': [],
            'error_queries': [],
        }
        self.report_lines = []

    def log(self, message: str, also_print: bool = True):
        """Log message to report and optionally print."""
        self.report_lines.append(message)
        if also_print:
            print(message)

    def initialize_chatbot(self) -> bool:
        """Initialize the chatbot with data."""
        self.log("\n" + "=" * 70)
        self.log("INITIALIZING CHATBOT")
        self.log("=" * 70)

        if not Path(self.root_folder).exists():
            self.log(f"ERROR: Root folder not found: {self.root_folder}")
            self.log("Please update ROOT_FOLDER in test_runner.py")
            return False

        self.log(f"Root folder: {self.root_folder}")

        try:
            start = time.time()
            self.chatbot = MappingChatBot(
                self.root_folder,
                use_parallel=True,
                max_workers=8,
                cache_enabled=True,
                cache_dir="./cache",
                cache_size_mb=500,
                cache_ttl_hours=24
            )
            self.chatbot.load_all_mappings()
            load_time = time.time() - start

            self.log(f"Loaded {len(self.chatbot.mappings_data)} vendors in {load_time:.2f}s")

            if not self.chatbot.mappings_data:
                self.log("WARNING: No data loaded!")
                return False

            return True

        except Exception as e:
            self.log(f"ERROR initializing chatbot: {str(e)}")
            return False

    def run_section(self, section_name: str, queries: list) -> dict:
        """Run a section of test queries."""
        section_results = {
            'total': len(queries),
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'details': []
        }

        self.log(f"\n{'─' * 70}")
        self.log(f"SECTION: {section_name} ({len(queries)} queries)")
        self.log('─' * 70)

        for i, (description, query) in enumerate(queries, 1):
            self.results['total'] += 1

            try:
                start = time.time()
                response = self.chatbot.process_query(query)
                elapsed = time.time() - start

                # Determine if test passed
                # A query passes if it returns a non-empty response without error markers
                has_results = "Found" in response or "##" in response or "**" in response
                is_no_match = "No mappings found" in response
                is_command_response = any(cmd in response for cmd in ["Help", "Statistics", "Cache", "Available Data", "cleared"])
                is_error = response.startswith("* Error") or "ERROR" in response.upper()

                # Pass conditions:
                # 1. Has results (found mappings)
                # 2. Is a valid "no match" response
                # 3. Is a valid command response (help, stats, etc.)
                passed = (has_results or is_no_match or is_command_response) and not is_error

                if passed:
                    section_results['passed'] += 1
                    self.results['passed'] += 1
                    status = "✓ PASS"
                else:
                    section_results['failed'] += 1
                    self.results['failed'] += 1
                    status = "✗ FAIL"
                    self.results['failed_queries'].append({
                        'section': section_name,
                        'description': description,
                        'query': query,
                        'response_preview': response[:200]
                    })

                self.log(f"  [{i:2d}] {status} ({elapsed:.3f}s) {description}")

                section_results['details'].append({
                    'description': description,
                    'query': query,
                    'status': 'PASS' if passed else 'FAIL',
                    'time': elapsed,
                    'response_preview': response[:100]
                })

            except Exception as e:
                section_results['errors'] += 1
                self.results['errors'] += 1
                self.log(f"  [{i:2d}] ✗ ERROR {description}: {str(e)}")
                self.results['error_queries'].append({
                    'section': section_name,
                    'description': description,
                    'query': query,
                    'error': str(e)
                })

        self.results['sections'][section_name] = section_results
        return section_results

    def run_all_tests(self):
        """Run all test sections."""
        self.log("\n" + "=" * 70)
        self.log("RAFM CHATBOT - AUTOMATED TEST SUITE")
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)

        # Initialize chatbot
        if not self.initialize_chatbot():
            self.log("\nTest aborted: Could not initialize chatbot")
            return self.results

        # Define all test sections
        test_sections = [
            ("1. Single Filter Queries", SINGLE_FILTER_QUERIES),
            ("2. Two Filter Combinations", TWO_FILTER_QUERIES),
            ("3. Three Filter Combinations", THREE_FILTER_QUERIES),
            ("4. Four Filter Combinations", FOUR_FILTER_QUERIES),
            ("5. Five Filter Combinations", FIVE_FILTER_QUERIES),
            ("6. All Six Filters", ALL_FILTER_QUERIES),
            ("7. Alternative Phrasing Styles", ALTERNATIVE_PHRASING_QUERIES),
            ("8. Special Commands", SPECIAL_COMMANDS),
            ("9. Edge Cases", EDGE_CASE_QUERIES),
        ]

        # Run each section
        total_start = time.time()
        for section_name, queries in test_sections:
            self.run_section(section_name, queries)
        total_time = time.time() - total_start

        # Print summary
        self.print_summary(total_time)

        # Save report
        self.save_report()

        return self.results

    def print_summary(self, total_time: float):
        """Print test summary."""
        self.log("\n" + "=" * 70)
        self.log("TEST SUMMARY")
        self.log("=" * 70)

        self.log(f"\nTotal Tests:    {self.results['total']}")
        self.log(f"Passed:         {self.results['passed']} ✓")
        self.log(f"Failed:         {self.results['failed']} ✗")
        self.log(f"Errors:         {self.results['errors']} !")

        if self.results['total'] > 0:
            success_rate = (self.results['passed'] / self.results['total']) * 100
            self.log(f"\nSuccess Rate:   {success_rate:.1f}%")

        self.log(f"Total Time:     {total_time:.2f}s")

        # Section breakdown
        self.log("\n" + "─" * 70)
        self.log("SECTION BREAKDOWN")
        self.log("─" * 70)

        for section_name, section_data in self.results['sections'].items():
            passed = section_data['passed']
            total = section_data['total']
            pct = (passed / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            self.log(f"  {section_name:<40} {passed:2d}/{total:2d} [{bar}] {pct:5.1f}%")

        # Failed queries
        if self.results['failed_queries']:
            self.log("\n" + "─" * 70)
            self.log(f"FAILED QUERIES ({len(self.results['failed_queries'])})")
            self.log("─" * 70)
            for i, fq in enumerate(self.results['failed_queries'][:10], 1):
                self.log(f"\n  {i}. [{fq['section']}] {fq['description']}")
                self.log(f"     Query: {fq['query'][:60]}...")
                self.log(f"     Response: {fq['response_preview'][:80]}...")

            if len(self.results['failed_queries']) > 10:
                self.log(f"\n  ... and {len(self.results['failed_queries']) - 10} more failures")

        # Error queries
        if self.results['error_queries']:
            self.log("\n" + "─" * 70)
            self.log(f"ERROR QUERIES ({len(self.results['error_queries'])})")
            self.log("─" * 70)
            for i, eq in enumerate(self.results['error_queries'][:5], 1):
                self.log(f"\n  {i}. [{eq['section']}] {eq['description']}")
                self.log(f"     Query: {eq['query'][:60]}...")
                self.log(f"     Error: {eq['error']}")

        self.log("\n" + "=" * 70)
        self.log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)

    def save_report(self):
        """Save the test report to a file."""
        report_file = "test_report.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.report_lines))
            print(f"\nReport saved to: {report_file}")
        except Exception as e:
            print(f"Could not save report: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("RAFM CHATBOT - AUTOMATED TEST RUNNER")
    print("=" * 70)

    # Check if root folder is configured
    if not Path(ROOT_FOLDER).exists():
        print(f"\nERROR: Root folder not found!")
        print(f"Path: {ROOT_FOLDER}")
        print("\nPlease update ROOT_FOLDER in test_runner.py to your data folder path.")
        print("Or set the MAPPING_ROOT_FOLDER environment variable.")
        return

    # Run tests
    runner = TestRunner(ROOT_FOLDER)
    results = runner.run_all_tests()

    # Return exit code based on results
    if results['failed'] > 0 or results['errors'] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()