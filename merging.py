import pandas as pd
import glob
import os

def merge_ddinter_evidence():
    print("Locating DDInter CSV files...")
    
    # Point this to wherever you saved the downloaded files
    # Make sure your files are actually in data/raw/
    file_path_pattern = "ddinter_downloads_code_*.csv" 
    
    # Note: Adjust the relative path depending on where you run the script from. 
    # If you run it from the root directory, use "data/raw/ddinter_downloads_code_*.csv"
    file_paths = glob.glob(file_path_pattern)
    
    if not file_paths:
        print("Error: No files found! Please check the folder path.")
        return
        
    print(f"Found {len(file_paths)} files. Merging now...")
    
    all_dfs = []
    
    for path in file_paths:
        try:
            # Read each CSV
            df = pd.read_csv(path)
            all_dfs.append(df)
            print(f"Loaded: {os.path.basename(path)} ({len(df)} records)")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            
    # Concatenate all dataframes together
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # CRITICAL: Drop duplicates! 
    # If Drug A (Cardiovascular) interacts with Drug B (Respiratory), 
    # that exact same interaction might be listed in BOTH the Code_C and Code_R files. 
    # We only want it in our final database once.
    initial_count = len(master_df)
    master_df = master_df.drop_duplicates(subset=['Drug_A', 'Drug_B'])
    final_count = len(master_df)
    
    print(f"\nDropped {initial_count - final_count} duplicate cross-category interactions.")
    
    # Ensure the output directory exists
    output_dir = "../../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final unified file
    output_path = os.path.join(output_dir, "ddinter_evidence.csv")
    master_df.to_csv(output_path, index=False)
    
    print(f"SUCCESS! Master evidence file saved to: {output_path}")
    print(f"Total unique clinical pairs ready for the UI: {final_count}")

if __name__ == "__main__":
    merge_ddinter_evidence()