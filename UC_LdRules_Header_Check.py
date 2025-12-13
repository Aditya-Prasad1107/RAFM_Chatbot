import os
import pandas as pd

def sanity_check(sources_folder):
    # required_headers = ["ID", "Dim./ Meas.", "Name", "Mapping (from layout)"]
    required_headers = ["ID", "Dim./ Meas. / CD", "Name", "CDR Detail Data"]

    print(f"\nRunning sanity check on: {sources_folder}\n{'='*60}\n")

    for source_name in os.listdir(sources_folder):
        source_path = os.path.join(sources_folder, source_name)
        if not os.path.isdir(source_path):
            continue

        # --- Check each Vendor under the Source ---
        for vendor_name in os.listdir(source_path):
            vendor_path = os.path.join(source_path, vendor_name)
            if not os.path.isdir(vendor_path):
                continue

            # --- Check if LdRules folder exists ---
            vendor_subfolders = {f.lower(): f for f in os.listdir(vendor_path)
                                 if os.path.isdir(os.path.join(vendor_path, f))}

            if "ldrules" not in vendor_subfolders:
                print(f"{source_name} -> {vendor_name} -> LdRules folder missing")
                continue

            # Get actual folder name (preserving case)
            ldrules_path = os.path.join(vendor_path, vendor_subfolders["ldrules"])

            # --- Check each Excel file in LdRules ---
            excel_files = [
                f for f in os.listdir(ldrules_path)
                if f.lower().endswith(('.xlsx', '.xls'))
            ]

            if not excel_files:
                print(f"{source_name} -> {vendor_name} -> LdRules -> No Excel files found")
                continue

            for excel_file in excel_files:
                excel_file_path = os.path.join(ldrules_path, excel_file)

                try:
                    # Read first 5 rows to check for headers
                    df = pd.read_excel(excel_file_path, header=None, nrows=5)
                    headers_found = False

                    for _, row in df.iterrows():
                        row_values = [str(cell).strip() for cell in row if pd.notna(cell)]

                        # Check for all required headers
                        if all(h in row_values for h in required_headers):
                            headers_found = True
                            break

                    if not headers_found:
                        print(f"{source_name} -> {vendor_name} -> LdRules -> {excel_file}")

                except Exception as e:
                    print(f"{source_name} -> {vendor_name} -> LdRules -> {excel_file} (Error: {e})")

    print("\nSanity check complete.\n")


file = r"C:\Users\aditya.prasad\OneDrive - Mobileum\Documents\OneDrive - Mobileum\Template Hierarchy Structure - Templates\FMS\Generic"
sanity_check(file)