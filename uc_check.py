import os
import pandas as pd

def sanity_check(sources_folder):
    required_vendor_folders = {"consolidated template", "ldrules"}
    required_headers = ["ID", "Dim./ Meas.", "Name", "Mapping (from layout)"]
    # required_headers = ["ID", "Dim./ Meas. / CD", "Name", "CDR Detail Data"]

    print(f"\nRunning sanity check on: {sources_folder}\n{'='*60}\n")

    for source_name in os.listdir(sources_folder):
        source_path = os.path.join(sources_folder, source_name)
        if not os.path.isdir(source_path):
            continue

        # --- Check if Source itself directly has vendor folders ---
        source_subfolders = [f.lower() for f in os.listdir(source_path)
                             if os.path.isdir(os.path.join(source_path, f))]

        if any(f in required_vendor_folders for f in source_subfolders):
            print(f"{source_name} - {{ Vendor missing (found 'Consolidated Template' or 'LdRules' directly under Source) }}")
            continue

        # --- Otherwise, check each Vendor under the Source ---
        for vendor_name in os.listdir(source_path):
            vendor_path = os.path.join(source_path, vendor_name)
            if not os.path.isdir(vendor_path):
                continue

            full_name = f"{source_name}/{vendor_name}"
            errors = []

            # --- Check 1: Required folders exist ---
            vendor_subfolders = [f.lower() for f in os.listdir(vendor_path)
                                 if os.path.isdir(os.path.join(vendor_path, f))]
            missing_folders = required_vendor_folders - set(vendor_subfolders)
            if missing_folders:
                errors.append(f"Missing folders: {', '.join(missing_folders)}")

            # Only proceed with Excel-related checks if Consolidated Template exists
            if "consolidated template" in vendor_subfolders:
                cons_temp_path = os.path.join(vendor_path,
                                              next(f for f in os.listdir(vendor_path)
                                                   if f.lower() == "consolidated template"))

                # --- Check 2: Exactly one Excel file ---
                excel_files = [
                    f for f in os.listdir(cons_temp_path)
                    if f.lower().endswith(('.xlsx', '.xls'))
                ]
                if len(excel_files) == 0:
                    errors.append("No Excel file found in 'Consolidated Template'")
                elif len(excel_files) > 1:
                    errors.append(f"Multiple Excel files found in 'Consolidated Template': {len(excel_files)}")
                else:
                    excel_file_path = os.path.join(cons_temp_path, excel_files[0])

                    # --- Check 3: File must have only one sheet ---
                    try:
                        excel_file = pd.ExcelFile(excel_file_path)
                        if len(excel_file.sheet_names) != 1:
                            errors.append(f"Excel file must have only 1 sheet (found {len(excel_file.sheet_names)})")
                        else:
                            # --- Check 4: Headers within first 5 rows (and no duplicates) ---
                            df = pd.read_excel(excel_file_path, header=None, nrows=5)
                            headers_found = False
                            duplicate_headers = False

                            for _, row in df.iterrows():
                                row_values = [str(cell).strip() for cell in row if pd.notna(cell)]

                                # Check for all required headers
                                if all(h in row_values for h in required_headers):
                                    headers_found = True
                                    for h in required_headers:
                                        if row_values.count(h) > 1:
                                            duplicate_headers = True
                                    break

                            if not headers_found:
                                errors.append("Required headers not found within first 5 rows")
                            elif duplicate_headers:
                                errors.append("One or more required headers appear more than once")

                    except Exception as e:
                        errors.append(f"Error reading Excel file: {e}")

            # --- Print results ---
            if errors:
                print(f"{full_name} - {{ {'; '.join(errors)} }}")

    print("\nSanity check complete.\n")

file = r"C:\Users\aditya.prasad\OneDrive - Mobileum\Documents\OneDrive - Mobileum\Tejas N's files - Templates\FMS\Generic"
sanity_check(file)