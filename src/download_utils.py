"""
Download and File Management Utilities

This module contains functions for downloading files, extracting archives,
and organizing permutation data.
"""

import os
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import requests


def download_file(url, filepath):
    """
    Download a file from URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    filepath : Path
        Local path to save the file
    
    Returns:
    --------
    filepath : Path
        Path to the downloaded file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Download completed: {filepath}")
    return filepath


def extract_zip(zip_path, extract_to):
    """
    Extract zip file with progress tracking.
    
    Parameters:
    -----------
    zip_path : Path
        Path to zip file
    extract_to : Path
        Directory to extract to
    """
    print(f"Extracting: {zip_path}")
    print(f"To directory: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive")
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)
    
    print(f"✓ Extraction completed to: {extract_to}")


def organize_permutations(source_dir, target_dir):
    """
    Organize downloaded permutations into the target directory.
    
    Parameters:
    -----------
    source_dir : Path
        Directory containing extracted permutations
    target_dir : Path
        Target permutations directory
    """
    print(f"Organizing permutations...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    if not source_dir.exists():
        print(f"✗ Source directory not found: {source_dir}")
        return
    
    # Look for permutation directories or files
    permutation_items = []
    for item in source_dir.rglob("*"):
        if item.is_dir() and (
            "permutation" in item.name.lower() or 
            item.name.endswith(".hetmat") or
            item.name.isdigit()
        ):
            permutation_items.append(item)
    
    print(f"Found {len(permutation_items)} potential permutation items")
    
    # Copy or move items to target directory
    for item in permutation_items[:5]:  # Show first 5 as example
        print(f"Found: {item.relative_to(source_dir)}")
    
    if len(permutation_items) > 5:
        print(f"... and {len(permutation_items) - 5} more")
    
    # Automatically proceed with organization (removed interactive prompt for papermill)
    if permutation_items:
        print("Proceeding with organizing permutations...")
        for item in tqdm(permutation_items, desc="Organizing"):
            target_path = target_dir / item.name
            if not target_path.exists():
                if item.is_dir():
                    shutil.copytree(item, target_path)
                else:
                    shutil.copy2(item, target_path)
        print(f"✓ Organized {len(permutation_items)} permutation items")
    else:
        print("No permutation items found to organize.")


def download_hetionet_permutations(download_url, data_dir, permutations_dir, hetio_permutations_dir):
    """
    Complete workflow to download and organize Hetionet permutations.
    
    Parameters:
    -----------
    download_url : str
        URL to download permutations from
    data_dir : Path
        Base data directory
    permutations_dir : Path
        Permutations directory
    hetio_permutations_dir : Path
        Target directory for hetio permutations
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Setup directories
        download_dir = data_dir / "downloads"
        zip_filename = "hetionet-v1.0-permutations.zip"
        zip_path = download_dir / zip_filename
        
        # Create directories
        download_dir.mkdir(parents=True, exist_ok=True)
        permutations_dir.mkdir(parents=True, exist_ok=True)
        hetio_permutations_dir.mkdir(parents=True, exist_ok=True)
        
        # Download if not exists
        if not zip_path.exists():
            download_file(download_url, zip_path)
        else:
            file_size = zip_path.stat().st_size
            print(f"✓ File already exists: {zip_path.name} ({file_size / (1024*1024):.1f} MB)")
        
        # Extract if not already extracted
        extract_check_path = download_dir / "hetionet-v1.0-permutations"
        if not extract_check_path.exists():
            extract_zip(zip_path, download_dir)
        else:
            print(f"✓ Archive already extracted at: {extract_check_path}")
        
        # Organize permutations
        organize_permutations(extract_check_path, hetio_permutations_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Download workflow failed: {e}")
        return False
