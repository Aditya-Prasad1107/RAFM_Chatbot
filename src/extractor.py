"""
Field extraction from folder structure with caching support.
Properly tracks which filename each expression comes from.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from openpyxl import load_workbook
from src.cache import get_cache, ExcelCache


def extract_combined_field_mappings_from_folder(
    folder_path: str,
    left_col: Union[str, int] = "B",
    field_col: Union[str, int] = "C",
    value_col: Union[str, int] = "D",
    sheet: Union[str, int] = 0,
    header_rows_to_skip: int = 2,
    strip_key_parts: bool = True,
    cache: Optional[ExcelCache] = None,
    use_cache: bool = True
) -> Tuple[
    Dict[Tuple[str, str], List[str]],           # mappings: (dim, field) -> [expressions]
    Dict[Tuple[str, str], List[str]],           # filename_info: (dim, field) -> [unique filenames]
    Dict[Tuple[str, str], List[str]]            # expression_filenames: (dim, field) -> [filename per expression]
]:
    """
    Extracts mappings from all Excel files in a folder.

    For each file:
      Key = (column B, column C)
      Value = list of column D entries aggregated across all files.

    IMPORTANT: Each expression is tracked with its source filename so we know
    which file each expression came from.

    Args:
        folder_path: Path to folder containing Excel files (typically 4-level-structure/LdRules)
        left_col: Column for left part of key (default "B" for Dim./Meas.)
        field_col: Column for field name (default "C" for Name)
        value_col: Column for mapping value (default "D" for Mapping)
        sheet: Sheet index or name (default 0)
        header_rows_to_skip: Number of header rows to skip (default 2)
        strip_key_parts: Whether to strip whitespace from keys (default True)
        cache: Optional cache instance to use for caching
        use_cache: Whether to use caching (default True)

    Returns:
        Tuple of three dictionaries:
        - mappings: (dim, field) tuples to lists of mapping expressions
        - filename_info: (dim, field) tuples to lists of unique filenames
        - expression_filenames: (dim, field) tuples to lists of filenames (one per expression, in order)
    """

    def col_to_index(col: Union[str, int]) -> int:
        """Convert Excel column letters (A, B, C...) to numeric indices."""
        if isinstance(col, str):
            num = 0
            for ch in col.upper():
                if not ('A' <= ch <= 'Z'):
                    raise ValueError(f"Invalid column letter: {col}")
                num = num * 26 + (ord(ch) - ord('A') + 1)
            return num
        elif isinstance(col, int) and col > 0:
            return col
        else:
            raise ValueError(
                "Column must be a positive integer or Excel column letter (A, B, C...)."
            )

    def norm(cell_value):
        """Normalize and safely convert cell values."""
        if cell_value is None:
            return ""
        v_str = (
            cell_value.strip()
            if (strip_key_parts and isinstance(cell_value, str))
            else str(cell_value)
        )
        return v_str.strip()

    left_idx = col_to_index(left_col)
    field_idx = col_to_index(field_col)
    value_idx = col_to_index(value_col)

    folder_obj = Path(folder_path)
    if not folder_obj.exists() or not folder_obj.is_dir():
        raise NotADirectoryError(f"Invalid folder path: {folder_path}")

    # Initialize cache if not provided and caching is enabled
    if cache is None and use_cache:
        cache = get_cache()

    # Result dictionaries
    result: Dict[Tuple[str, str], List[str]] = {}
    filename_info: Dict[Tuple[str, str], List[str]] = {}
    expression_filenames: Dict[Tuple[str, str], List[str]] = {}

    # Only process .xlsx files
    excel_files = [
        f for f in folder_obj.iterdir()
        if f.is_file() and f.suffix.lower() == ".xlsx" and not f.name.startswith('~$')
    ]

    if not excel_files:
        print(f"No Excel files found in the folder: {folder_path}")
        return result, filename_info, expression_filenames

    print(f"Processing {len(excel_files)} Excel file(s) from folder: {folder_path}")

    # Track cache performance
    cache_hits = 0
    cache_misses = 0
    files_processed = 0

    for file_path in excel_files:
        try:
            # Prepare extraction parameters for cache key
            extraction_params = {
                'left_col': left_col,
                'field_col': field_col,
                'value_col': value_col,
                'sheet': sheet,
                'header_rows_to_skip': header_rows_to_skip,
                'strip_key_parts': strip_key_parts
            }

            # Data for this specific file
            file_data: Dict[Tuple[str, str], List[str]] = {}
            file_filename_info: Dict[Tuple[str, str], List[str]] = {}
            file_expression_filenames: Dict[Tuple[str, str], List[str]] = {}

            # Try to get from cache first
            cached_data = None
            if use_cache and cache:
                cached_data = cache.get_cached_extraction_with_filenames(file_path, extraction_params)
                if cached_data:
                    if isinstance(cached_data, tuple) and len(cached_data) == 3:
                        file_data, file_filename_info, file_expression_filenames = cached_data
                        cache_hits += 1
                    else:
                        # Invalid cache format, process file
                        cached_data = None
                        cache_misses += 1
                else:
                    cache_misses += 1

            # If not in cache, process the file
            if cached_data is None:
                files_processed += 1
                current_filename = file_path.stem  # Filename without extension

                wb = load_workbook(file_path, data_only=True, read_only=True)

                # Select target sheet
                if isinstance(sheet, int):
                    if sheet < 0 or sheet >= len(wb.sheetnames):
                        print(f"Skipping {file_path.name} - sheet index out of range.")
                        wb.close()
                        continue
                    ws = wb[wb.sheetnames[sheet]]
                else:
                    if sheet not in wb.sheetnames:
                        print(f"Skipping {file_path.name} - sheet '{sheet}' not found.")
                        wb.close()
                        continue
                    ws = wb[sheet]

                for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
                    if i <= header_rows_to_skip:
                        continue
                    if max(left_idx, field_idx, value_idx) - 1 >= len(row):
                        continue

                    left_val = norm(row[left_idx - 1])
                    field_val = norm(row[field_idx - 1])
                    value_val = norm(row[value_idx - 1])

                    # Skip if any required column is empty
                    if not left_val or not field_val or not value_val:
                        continue

                    # Skip if value column is just "-", "NA", or "N/A"
                    if value_val.strip() in ("-", 'NA', 'N/A'):
                        continue

                    # Filter out #REF! and other Excel errors at the source
                    invalid_patterns = [
                        '#ref!',
                        '#n/a',
                        '#value!',
                        '#name?',
                        '#null!',
                        '#div/0!',
                    ]

                    value_lower = value_val.lower()
                    is_invalid = any(pattern in value_lower for pattern in invalid_patterns)

                    if is_invalid:
                        continue

                    key = (left_val, field_val)

                    # Add expression to file_data
                    if key not in file_data:
                        file_data[key] = []
                    file_data[key].append(value_val)

                    # Track unique filenames for this key
                    if key not in file_filename_info:
                        file_filename_info[key] = []
                    if current_filename not in file_filename_info[key]:
                        file_filename_info[key].append(current_filename)

                    # Track filename for each expression (in order)
                    if key not in file_expression_filenames:
                        file_expression_filenames[key] = []
                    file_expression_filenames[key].append(current_filename)

                wb.close()

                # Save to cache
                if use_cache and cache:
                    cache.save_extraction_result_with_filenames(
                        file_path,
                        extraction_params,
                        (file_data, file_filename_info, file_expression_filenames)
                    )

                print(f"Processed: {file_path.name} ({len(file_data)} mappings)")

            # Merge file data into global results
            for key, values in file_data.items():
                if key not in result:
                    result[key] = []
                result[key].extend(values)

            # Merge filename info (unique filenames per key)
            for key, filenames in file_filename_info.items():
                if key not in filename_info:
                    filename_info[key] = []
                for fname in filenames:
                    if fname not in filename_info[key]:
                        filename_info[key].append(fname)

            # Merge expression filenames (one filename per expression, in order)
            for key, filenames in file_expression_filenames.items():
                if key not in expression_filenames:
                    expression_filenames[key] = []
                expression_filenames[key].extend(filenames)

        except Exception as e:
            print(f"Error reading file '{file_path.name}': {e}")
            continue

    # Print cache statistics
    if use_cache and cache:
        print(f"\nCache Performance:")
        print(f"   Cache Hits: {cache_hits}")
        print(f"   Cache Misses: {cache_misses}")
        print(f"   Files Actually Processed: {files_processed}")
        if cache_hits + cache_misses > 0:
            hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
            print(f"   Hit Rate: {hit_rate:.1f}%")

    print(f"Completed processing {len(excel_files)} file(s). Found {len(result)} unique keys.")

    return result, filename_info, expression_filenames