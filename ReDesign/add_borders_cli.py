# ReDesign/add_borders_cli.py
"""
Add pink element borders to existing episode reconstructed images.

Usage:
    python -m ReDesign.add_borders_cli --episode <episode_dir>
    python -m ReDesign.add_borders_cli --episode src/agent_output/episodes/Figma_10_20260116_101448
    
    # Multiple episodes
    python -m ReDesign.add_borders_cli --batch src/agent_output/episodes/
"""
from __future__ import annotations
from pathlib import Path
import argparse

from .reconstruction import add_borders_to_existing_episode


def main():
    parser = argparse.ArgumentParser(
        description="Add pink element borders to existing episode reconstructed images"
    )
    parser.add_argument(
        "--episode", "-e",
        type=str,
        help="Path to a single episode directory to process (e.g. src/agent_output/episodes/<episode_name>)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Path to a parent directory containing multiple episode directories; every subdirectory that has a parse.json file is processed"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="reconstructed_bordered.png",
        help="Output image filename written inside each episode directory (default: reconstructed_bordered.png)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce console output verbosity during processing"
    )
    
    args = parser.parse_args()
    
    if args.episode:
        # Single episode
        bordered_path = add_borders_to_existing_episode(
            episode_dir=args.episode,
            output_name=args.output,
            verbose=not args.quiet
        )
        print(f"\n✅ Bordered image saved to: {bordered_path}")
        
    elif args.batch:
        # Batch mode - process all episode directories
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"❌ Batch directory not found: {batch_path}")
            return
        
        # Find all episode directories (those with parse.json)
        episode_dirs = [
            d for d in batch_path.iterdir()
            if d.is_dir() and (d / "parse.json").exists()
        ]
        
        print(f"Found {len(episode_dirs)} episode directories")
        
        success_count = 0
        for i, episode_dir in enumerate(episode_dirs):
            print(f"\n[{i+1}/{len(episode_dirs)}] Processing: {episode_dir.name}")
            try:
                add_borders_to_existing_episode(
                    episode_dir=str(episode_dir),
                    output_name=args.output,
                    verbose=not args.quiet
                )
                success_count += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print(f"\n✅ Completed: {success_count}/{len(episode_dirs)} episodes")
    
    else:
        parser.print_help()
        print("\n❌ Please provide --episode or --batch argument")


if __name__ == "__main__":
    main()