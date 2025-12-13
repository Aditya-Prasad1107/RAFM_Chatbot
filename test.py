import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_source(source_path):
    return [vendor for vendor in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, vendor))]


def list_folders_hierarchy(base_path):
    hierarchy = {}
    future_to_info = {}

    with ThreadPoolExecutor() as executor:
        for domain in os.listdir(base_path):
            domain_path = os.path.join(base_path, domain)
            if os.path.isdir(domain_path):
                hierarchy[domain] = {}

                for module in os.listdir(domain_path):
                    module_path = os.path.join(domain_path, module)
                    if os.path.isdir(module_path):
                        hierarchy[domain][module] = {}

                        for source in os.listdir(module_path):
                            source_path = os.path.join(module_path, source)
                            if os.path.isdir(source_path):
                                future = executor.submit(process_source, source_path)
                                future_to_info[future] = (domain, module, source)

        for future in as_completed(future_to_info):
            domain, module, source = future_to_info[future]
            vendors = future.result()
            hierarchy[domain][module][source] = vendors

    return hierarchy


# Example usage:
base_path = r"C:\Users\aditya.prasad\OneDrive - Mobileum\Documents\OneDrive - Mobileum\Template Hierarchy Structure - Templates"
folder_hierarchy = list_folders_hierarchy(base_path)

# Print the hierarchy in a structured way
# for domain, modules in folder_hierarchy.items():
#     print(f"Domain: {domain}")
#     for module, sources in modules.items():
#         print(f"  Module: {module}")
#         for source, vendors in sources.items():
#             print(f"    Source: {source}")
#             for vendor in vendors:
#                 if vendor not in ("Layout", "LdRules"):
#                     print(f"      Vendor: {vendor}")


# After collecting the hierarchy with the previous function

# Print only modules and their sources
for domain, modules in folder_hierarchy.items():
    print(f"Domain: {domain}")
    for module, sources in modules.items():
        print(f"  Module: {module}")
        for source in sources:
            print(f"    Source: {source}")

