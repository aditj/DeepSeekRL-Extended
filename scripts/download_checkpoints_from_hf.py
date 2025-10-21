#!/usr/bin/env python3
"""
Script to download model checkpoints from Hugging Face Hub.

Usage:
    python download_checkpoints_from_hf.py --repo_id <org/repo_name> --output_dir <path> [options]

Example:
    # Download a single repo (all revisions)
    python download_checkpoints_from_hf.py \
        --repo_id username/my-model \
        --output_dir ./models/my-model
    
    # Download specific checkpoints only
    python download_checkpoints_from_hf.py \
        --repo_id username/my-model \
        --output_dir ./models/my-model \
        --revisions checkpoint_5000 checkpoint_10000
    
    # Download multiple repos by pattern
    python download_checkpoints_from_hf.py \
        --repo_pattern "username/multi_task_rl_llms-*" \
        --output_dir ./models \
        --list_repos  # First, list what would be downloaded

    # Download multiple repos by pattern and download all revisions
    python download_checkpoints_from_hf.py \
        --repo_pattern "username/multi_task_rl_llms-*" \
        --output_dir ./models \
        --all_revisions
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
import re

try:
    from huggingface_hub import HfApi, snapshot_download, list_repo_refs
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)


def list_user_repos(username: str, pattern: Optional[str] = None, token: Optional[str] = None) -> List[str]:
    """
    List all repositories for a user, optionally filtered by pattern.
    
    Args:
        username: HuggingFace username or organization
        pattern: Optional glob pattern to filter repos (e.g., "prefix-*")
        token: HuggingFace API token
    
    Returns:
        List of repository IDs
    """
    api = HfApi(token=token)
    
    try:
        # List all models for the user
        models = api.list_models(author=username)
        repo_ids = [model.id for model in models]
        
        # Filter by pattern if provided
        if pattern:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            repo_ids = [repo for repo in repo_ids if re.match(regex_pattern, repo)]
        
        return sorted(repo_ids)
    except Exception as e:
        print(f"Error listing repositories: {e}")
        return []


def get_repo_revisions(repo_id: str, token: Optional[str] = None) -> List[str]:
    """
    Get all branches/revisions for a repository.
    
    Args:
        repo_id: Repository ID
        token: HuggingFace API token
    
    Returns:
        List of revision names
    """
    try:
        refs = list_repo_refs(repo_id, repo_type="model", token=token)
        
        # Get all branches (excluding main if you want only checkpoints)
        branches = [ref.name for ref in refs.branches]
        
        return sorted(branches)
    except Exception as e:
        print(f"Error getting revisions for {repo_id}: {e}")
        return []


def download_checkpoint(
    repo_id: str,
    revision: Optional[str] = None,
    output_dir: str = "./models",
    token: Optional[str] = None,
    resume: bool = True,
) -> bool:
    """
    Download a checkpoint from HuggingFace Hub.
    
    Args:
        repo_id: Repository ID on HuggingFace
        revision: Specific revision/branch to download (None for main)
        output_dir: Directory to save the checkpoint
        token: HuggingFace API token
        resume: Whether to resume interrupted downloads
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory structure
        output_path = Path(output_dir)
        
        if revision and revision != "main":
            # Save to repo_name/revision/ structure
            repo_name = repo_id.split('/')[-1]
            final_output = output_path / repo_name / revision
        else:
            # Save to repo_name/ for main branch
            repo_name = repo_id.split('/')[-1]
            final_output = output_path / repo_name / "main"
        
        print(f"\n{'='*80}")
        print(f"Repository: {repo_id}")
        print(f"Revision: {revision or 'main'}")
        print(f"Output: {final_output}")
        print(f"{'='*80}")
        
        # Download the model
        print("Downloading... (this may take a while)")
        
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=None,  # Use default cache
            local_dir=str(final_output),
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=resume,
            token=token,
        )
        
        print(f"✓ Successfully downloaded to: {final_output}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {repo_id}@{revision or 'main'}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all checkpoints from a specific repo
  python download_checkpoints_from_hf.py \\
      --repo_id username/my-model \\
      --output_dir ./models

  # Download only specific checkpoints
  python download_checkpoints_from_hf.py \\
      --repo_id username/my-model \\
      --output_dir ./models \\
      --revisions checkpoint_5000 checkpoint_10000

  # List all repos matching a pattern
  python download_checkpoints_from_hf.py \\
      --repo_pattern "username/multi_task_rl_llms-*" \\
      --list_repos

  # Download all repos matching a pattern
  python download_checkpoints_from_hf.py \\
      --repo_pattern "username/multi_task_rl_llms-*" \\
      --output_dir ./models \\
      --revisions checkpoint_10000

  # Download only main branch
  python download_checkpoints_from_hf.py \\
      --repo_id username/my-model \\
      --output_dir ./models \\
      --main_only
        """,
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        help="HuggingFace repository ID (format: username/repo-name)",
    )
    
    parser.add_argument(
        "--repo_pattern",
        type=str,
        help="Pattern to match multiple repositories (e.g., 'username/prefix-*')",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./downloaded_models",
        help="Directory to save downloaded checkpoints (default: ./downloaded_models)",
    )
    
    parser.add_argument(
        "--revisions",
        type=str,
        nargs="+",
        help="Specific revisions/branches to download (e.g., checkpoint_1000 checkpoint_5000)",
    )
    
    parser.add_argument(
        "--main_only",
        action="store_true",
        help="Download only the main branch (skip checkpoint branches)",
    )
    
    parser.add_argument(
        "--all_revisions",
        action="store_true",
        help="Download all available revisions/branches",
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (if not provided, will use huggingface-cli login)",
    )
    
    parser.add_argument(
        "--list_repos",
        action="store_true",
        help="List repositories matching the pattern without downloading",
    )
    
    parser.add_argument(
        "--list_revisions",
        action="store_true",
        help="List available revisions without downloading",
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume interrupted downloads (start fresh)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.repo_id and not args.repo_pattern:
        print("Error: Either --repo_id or --repo_pattern must be provided")
        sys.exit(1)
    
    if args.repo_id and args.repo_pattern:
        print("Error: Cannot use both --repo_id and --repo_pattern")
        sys.exit(1)
    
    # Handle repo pattern (multiple repos)
    if args.repo_pattern:
        # Extract username and pattern
        if '/' not in args.repo_pattern:
            print("Error: repo_pattern must be in format 'username/pattern'")
            sys.exit(1)
        
        username, pattern = args.repo_pattern.split('/', 1)
        
        print(f"Searching for repositories matching: {args.repo_pattern}")
        repos = list_user_repos(username, f"{username}/{pattern}", args.token)
        
        if not repos:
            print(f"No repositories found matching pattern: {args.repo_pattern}")
            sys.exit(1)
        
        print(f"\nFound {len(repos)} repository(s):")
        for i, repo in enumerate(repos, 1):
            print(f"  {i}. {repo}")
        
        if args.list_repos:
            sys.exit(0)
        
        # Process each repo
        if not args.dry_run:
            response = input(f"\nProceed with downloading {len(repos)} repositories? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Download cancelled.")
                sys.exit(0)
        
        total_successful = 0
        total_failed = 0
        
        for repo_id in repos:
            # Determine which revisions to download
            if args.main_only:
                revisions_to_download = [None]  # None means main branch
            elif args.revisions:
                revisions_to_download = args.revisions
            elif args.all_revisions:
                revisions_to_download = get_repo_revisions(repo_id, args.token)
                if not revisions_to_download:
                    revisions_to_download = [None]  # Fallback to main
            else:
                # Default: download main only
                revisions_to_download = [None]
            
            print(f"\n{'='*80}")
            print(f"Processing: {repo_id}")
            print(f"Revisions to download: {revisions_to_download if revisions_to_download != [None] else ['main']}")
            print(f"{'='*80}")
            
            if args.dry_run:
                print(f"[DRY RUN] Would download {len(revisions_to_download)} revision(s)")
                continue
            
            for revision in revisions_to_download:
                success = download_checkpoint(
                    repo_id=repo_id,
                    revision=revision,
                    output_dir=args.output_dir,
                    token=args.token,
                    resume=not args.no_resume,
                )
                
                if success:
                    total_successful += 1
                else:
                    total_failed += 1
        
        # Summary
        if not args.dry_run:
            print(f"\n{'='*80}")
            print("Download Summary:")
            print(f"  Successful: {total_successful}")
            print(f"  Failed: {total_failed}")
            print(f"  Total: {total_successful + total_failed}")
            print(f"{'='*80}")
    
    # Handle single repo
    else:
        repo_id = args.repo_id
        
        # List revisions if requested
        if args.list_revisions:
            print(f"Fetching revisions for {repo_id}...")
            revisions = get_repo_revisions(repo_id, args.token)
            
            if not revisions:
                print("No revisions found (or only main branch exists)")
            else:
                print(f"\nFound {len(revisions)} revision(s):")
                for i, rev in enumerate(revisions, 1):
                    print(f"  {i}. {rev}")
            sys.exit(0)
        
        # Determine which revisions to download
        if args.main_only:
            revisions_to_download = [None]  # None means main branch
        elif args.revisions:
            revisions_to_download = args.revisions
        elif args.all_revisions:
            revisions_to_download = get_repo_revisions(repo_id, args.token)
            if not revisions_to_download:
                revisions_to_download = [None]  # Fallback to main
        else:
            # Default: download main only
            revisions_to_download = [None]
        
        print(f"Repository: {repo_id}")
        print(f"Output directory: {args.output_dir}")
        print(f"Revisions to download: {revisions_to_download if revisions_to_download != [None] else ['main']}")
        
        if args.dry_run:
            print("\nDry run mode - no downloads will be performed.")
            sys.exit(0)
        
        response = input(f"\nProceed with downloading {len(revisions_to_download)} revision(s)? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Download cancelled.")
            sys.exit(0)
        
        successful = 0
        failed = 0
        
        for revision in revisions_to_download:
            success = download_checkpoint(
                repo_id=repo_id,
                revision=revision,
                output_dir=args.output_dir,
                token=args.token,
                resume=not args.no_resume,
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*80}")
        print("Download Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(revisions_to_download)}")
        print(f"{'='*80}")
        
        if successful > 0:
            print(f"\nModels downloaded to: {args.output_dir}")
        
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

